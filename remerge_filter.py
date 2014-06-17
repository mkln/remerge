"""
Loads the cleaned and geocoded PATSTAT and Amadeus files of the same country.
For every PATSTAT name, returns a list of 20 (or less) companies that have
the most similar names to the PATSTAT name.

Runs in parallel. Resource intensive.
"""

import sys
sys.path.append("/home/desktop/patstat_data/all_code/dbUtils/")
import pandas as pd
import re
import numpy as np
import consolidate_df as cd
import pandas as pd
import re
import string
import time
from jellyfish import jaro_winkler as jw
from dedupe.distance import haversine
import Levenshtein
from IPython.parallel import Client
import os
import gc
import company_legal_id as legal
import itertools as it
from ccolors import *
from smart_type_converter import as_str

eu27_abbrev = {'austria':'at',
               'bulgaria':'bg',
               'belgium':'be',
               'italy':'it',
               'uk':'gb',
               'france':'fr',
               'germany':'de',
               'slovakia':'sk',
               'sweden':'se',
               'portugal':'pt',
               'poland':'pl',
               'hungary':'hu',
               'ireland':'ie',
               'estonia':'ee',
               'spain':'es',
               'cyprus':'cy',
               'czech republic':'cz',
               'netherlands':'nl',
               'slovenia':'si',
               'romania':'ro',
               'denmark':'dk',
               'lithuania':'lt',
               'luxembourg':'lu',
               'latvia':'lv',
               'malta':'mt',
               'finland':'fi',
               'greece':'el',
               'greece':'gr'
               }


eu27 = ['sk', 
        'se', 'pt', 'pl', 'hu', 'ie', 'ee', 
        'cy', 'cz', 'nl', 'si', 'ro', 'dk', 'lt',
        'lu', 'lv', 'mt', 'fi', 'gr', 'at', 'bg', 'be', 'it', 'gb', 
        'de', 'es', 'fr'
        ]

abbrev_dict = {}

company_input_root = '/home/desktop/patstat_data/all_code/remerge/amadeus_wrds/'
company_output_root = '/home/desktop/patstat_data/all_code/remerge/best_matches/'
patstat_root = '/home/desktop/patstat_data/all_code/dbClean/patstat_geocoded/'

n_matches = 10

cores = 10 #sys.argv[1]
list_args = eu27

'''
file_header = ['', 'appln_id','person_id','person_name','person_address','person_ctry_code','applicant_seq','inventor_seq','coauthors','ipc_code','year']
dtypes = [np.int32, np.int32, np.int32, object, object, object, np.int32, np.int32, object, object, np.int32]

typedict = dict(zip(file_header, dtypes))
'''

def parallel_clear():
    dview.results.clear()
    par_client.results.clear()
    gc.collect()
    return

for country in eu27:
    print country
    company_file = company_input_root + 'cleaned_output_%s.tsv' % country
    patstat_file = patstat_root + 'patstat_geocoded_%s_v2.csv' % country

    df_amadeus = pd.read_csv(company_file, sep='\t', usecols = ['company_name', 'company_id'])
    df_amadeus = df_amadeus.drop_duplicates()

    df_patstat = pd.read_csv(patstat_file, sep='\t', usecols = ['person_name','person_id'])

    df_patstat = df_patstat.rename(columns = {'person_name': 'patstat_name',
                                              'person_id': 'patstat_id'})
                          
    print 'looking for best matches for %s ... ' % country

    output_file = company_output_root + 'best_matches_%s_wrds_v3.csv' % country

    print 'there are %d names in PATSTAT for this country.' % len(df_patstat)
    print 'there are %d companies in Amadeus for this country.' % len(df_amadeus)

    df_patstat['patstat_name'] = [n.strip() if isinstance(n, str) else None 
                                  for n in df_patstat['patstat_name']]
    
    df_amadeus['company_name'] = [n.strip() if isinstance(n, str) else None
                                  for n in df_amadeus['company_name']]

    df_patstat['first_n'] = [n.replace(' ','').strip()[:2] if isinstance(n, str) else None
                                for n in df_patstat['patstat_name']]
    df_amadeus['first_n'] = [n.replace(' ','').strip()[:2] if isinstance(n, str) else None
                                  for n in df_amadeus['company_name']]
    
    df_patstat = df_patstat.dropna()
    df_amadeus = df_amadeus.dropna()
    
    find_legal = legal.FindLegalId()

    start_time = time.time()
    print 'started'
    print bcolors.OKBLUE
    os.system('ipcluster start -n %d &' % cores)
    time.sleep(33)
    print bcolors.ENDC
    
    par_client = Client()
    dview = par_client[:]
    dview.block = True
    
    # Sync the necessary imports and path settings
    dview.execute('import sys')
    dview.execute('sys.path.append("/home/desktop/patstat_data/all_code/dbUtils/")')

    with dview.sync_imports():
        import jellyfish
        import pandas
        import Levenshtein
        import company_legal_id as legal
        
    dview.push({'country': country})  
    dview.push({'find_legal': find_legal})
    
    @dview.parallel(block=True)
    def separate_legal(name):
        """
        Looks for a legal identifier inside a name and takes it out.
        Takes a list of names (for parallel) and returns a tuple.
        """
        return find_legal.separate_comp_legalid(name, country, 'outside')
            
    # split company name into company name and legal id
    names_am = separate_legal.map(list(df_amadeus['company_name'])) 
    print 'separated names and legal (company)'
    df_amadeus['company_name'], df_amadeus['am_legal'] = it.izip(*names_am)
    parallel_clear()
                    
    # split patstat name into company name and legal id
    names_ps = separate_legal.map(list(df_patstat['patstat_name'])) 
    print 'separated names and legal (patstat)'
    df_patstat['patstat_name'], df_patstat['ps_legal'] = it.izip(*names_ps)
    parallel_clear()

    # create list of names for complex indexing
    df_patstat['name_split'] = [s.split(' ') if isinstance(s, str) else None
                              for s in df_patstat['patstat_name']]
    df_amadeus['name_split'] = [s.split(' ') if isinstance(s, str) else None
                                for s in df_amadeus['company_name']]

    # complex indexing patstat
    pidx = []
    for name_list in df_patstat['name_split']:
        pidx.append([v[:3] for v in name_list if len(v[:3])>1 ])

    pidx1 = []
    pidx2 = []
    for i,v in enumerate(pidx[:]):
        if len(v)==0:
            pidx1.append('')
            pidx2.append('')
        else:
            pidx1.append(v[0])
            if len(v)>1:
                pidx2.append(v[1])
            else:
                pidx2.append('')
                
    df_patstat['idx1_list'] = pidx1
    df_patstat['idx2_list'] = pidx2
    print 'patstat shape:', df_patstat.shape       
         
    # Amadeus

    aidx = []
    for name_list in df_amadeus['name_split']:
        aidx.append([v[:3] for v in name_list if len(v[:3])>1 ])

    aidx1 = []
    aidx2 = []
    for i,v in enumerate(aidx[:]):
        if len(v)==0:
            aidx1.append('')
            aidx2.append('')
        else:
            aidx1.append(v[0])
            if len(v)>1:
                aidx2.append(v[1])
            else:
                aidx2.append('')

    df_amadeus['idx1_list'] = aidx1
    df_amadeus['idx2_list'] = aidx2
    print 'company shape:', df_amadeus.shape
    
    df_amadeus['company_name'] = df_amadeus['company_name'].apply(as_str)
    
    df_patstat['patstat_name'] = df_patstat['patstat_name'].apply(as_str)
    df_patstat = df_patstat[df_patstat['patstat_name'].apply(len) > 1]
    
    df_patstat = df_patstat.drop('name_split', 1)
    df_amadeus = df_amadeus.drop('name_split', 1)

    df_amadeus = df_amadeus.set_index(['first_n','idx1_list','idx2_list'])

    start_time = time.time()
    
    @dview.parallel(block=True)
    def calc_dists(prow, redux=False):
        """
        Calculates the string distance between a PATSTAT name and company names.
        Performs the blocking strategy onthe company dataset to restrict
        the search and save on time and memory.
        """
        df_amadeus_temp = []
        
        if prow['first_n']:
            try:
                df_amadeus_temp.append(df_amadeus.xs(prow['first_n'], level = 'first_n'))  
            except:
                pass

        # add rows to look for matches, but only if
        # different than what we have and not empty
        
        if (prow['idx1_list'][:2] != prow['first_n']) and (prow['idx1_list']!=''):
            try:
                df_amadeus_temp.append(df_amadeus.xs(prow['idx1_list'], level = 'idx1_list')) # first word in amadeus name
            except:
                pass
            try:
                df_amadeus_temp.append(df_amadeus.xs(prow['idx1_list'], level = 'idx2_list'))  # second word in amadeus name
            except:
                pass
        if (prow['idx2_list'][:2] != prow['first_n']) and (prow['idx2_list']!='') and (prow['idx2_list']!=prow['idx1_list']):
            try:
                df_amadeus_temp.append(df_amadeus.xs(prow['idx2_list'], level = 'idx1_list'))
            except:
                pass
            try:
                df_amadeus_temp.append(df_amadeus.xs(prow['idx2_list'], level = 'idx2_list'))
            except:
                pass
        
        try:
            df_amadeus_temp = pandas.concat(df_amadeus_temp, ignore_index=True)
        except:
            return [[prow['patstat_id'], prow['patstat_name'], prow['ps_legal'], '', '', '', None, None]]
        
        matches = []
        res_list = []
        for jdx, arow in df_amadeus_temp.iterrows():
            try:
                lev_name_dist = Levenshtein.ratio(prow['patstat_name'], arow['company_name'])
            except:
                lev_name_dist = 0
                
            jw_name_dist = jellyfish.jaro_winkler(prow['patstat_name'], arow['company_name'])
            lev_name_dist = 1 - lev_name_dist
            jw_name_dist = 1 - jw_name_dist
            res_list.append([prow['patstat_id'], prow['patstat_name'], prow['ps_legal'],
                                 arow['company_id'], arow['company_name'], arow['am_legal'],
                                 lev_name_dist, jw_name_dist])

        sorted_dist = sorted(res_list, key=lambda x: (x[6]))
        patstat_company_match_lev = sorted_dist[:n_matches]    
        matches.extend(patstat_company_match_lev)

        sorted_dist = sorted(res_list, key=lambda x: (x[7]))
        patstat_company_match_jw = sorted_dist[:n_matches]
        matches.extend(patstat_company_match_jw)     
                           
        return matches
        
    dview.push({'n_matches': n_matches})    
    dview.push({'df_amadeus': df_amadeus})  
 
    # find company matches for patstat names
    print 'now looking for company good matches'
    string_dists = calc_dists.map([v for i,v in df_patstat.iterrows()])
    print 'found best matching company companies for patstat'
    parallel_clear()
    
    print bcolors.OKBLUE
    os.system('ipcluster stop &')
    time.sleep(5)
    print bcolors.ENDC
    
    # get single dataframe from parallel output
    print 'moving to a dataframe'
    p_a_matches = []
    for i in string_dists:
        p_a_matches.extend(i)
        
    df_out = pd.DataFrame(p_a_matches,
                      columns=['patstat_id', 'patstat_name', 'ps_legal', 
                               'company_id', 'company_name', 'am_legal',
                               'lev_name_dist', 'jw_name_dist'])
    # drop duplicates
    df_out = df_out.drop_duplicates()
    
    parallel_time = time.time() - start_time
    print parallel_time/len(df_patstat), 'per record'
    print 'total %s' % str(parallel_time)
    
    df_amadeus = pd.read_csv(company_file, sep='\t', usecols = ['company_id', 'lat', 'lng'])
    df_amadeus = df_amadeus.drop_duplicates()
    df_amadeus = df_amadeus[pd.notnull(df_amadeus['lat'])]

    df_patstat = pd.read_csv(patstat_file, sep='\t', usecols = ['person_id', 'lat', 'lng'])
    df_patstat = df_patstat[pd.notnull(df_patstat['lat'])]
    
    geodist = df_out[['patstat_id','company_id']].set_index('patstat_id').join(df_patstat.set_index('person_id'))
    geodist = geodist.rename(columns = {'lat':'plat', 'lng': 'plng'})
    geodist = geodist.reset_index().set_index('company_id').join(df_amadeus.set_index('company_id')).reset_index()
    geodist = geodist.rename(columns = {'index':'patstat_id','lat':'alat', 'lng': 'alng'})
    
    geodist = geodist[pd.notnull(geodist['plat']*geodist['alat'])]
    
    # geocode matches
    print 'computing geo-distances of selected matches'
    geo_dist = []
    for i,v in geodist.iterrows():
        geo_dist.append(haversine.compareLatLong((v['plat'],v['plng']),(v['alat'],v['alng'])))

    geodist['geo_dist'] = geo_dist
    geodist = geodist[['patstat_id','company_id','geo_dist']].set_index(['patstat_id','company_id'])
    df_out = df_out.set_index(['patstat_id','company_id']).join(geodist)
    
    print 'done with geo-distances'
    
    df_out['country'] = country
    
    # save
    print 'saving to file', output_file
    df_out.to_csv(output_file, sep='\t')
    print 'done!'
    
