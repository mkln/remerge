"""
Takes the .best matches (= candidates), 
          .PATSTAT cleaned and geocoded, 
          .Amadeus cleaned and geocoded 
files for a country, manipulates information to create regressors that
will be used when fitting the model to the training data.

This creates the variables for the whole data as well, to be used during
the final matching.
"""

import sys
sys.path.append("/home/desktop/patstat_data/all_code/dbUtils/")

import consolidate_df as cd
import pandas as pd
from IPython.parallel import Client
import bag_of_words as bofw
import company_legal_id as legal
import os
import gc
import time
import itertools as it
import jellyfish
import Levenshtein
import fuzzy
import re
from ccolors import *
import numpy as np
from smart_type_converter import as_str

countries = ['at', 'bg', 'be', 
        'it', 'gb', 'fr', 
        'de', 'sk', 'se', 'pt', 'pl',
        'hu', 'ie', 'ee', 'es', 'cy', 
         'nl', 'si', 'ro', 'dk', 
        'cz', 'lt','lu', 'lv', 'mt', 'fi', 'gr'
        ]

input_bestmatches = '/home/desktop/patstat_data/all_code/remerge/best_matches/'
input_company = '/home/desktop/patstat_data/all_code/remerge/amadeus_wrds/'
input_patstat = '/home/desktop/patstat_data/all_code/dbClean/patstat_geocoded/'
input_labels = '/home/desktop/patstat_data/all_code/remerge/labeling_data/'

output_root = '/home/desktop/patstat_data/all_code/remerge/regression_data/'

sector_matrix_file = '/home/desktop/patstat_data/all_code/remerge/labeled_sample/naics_ipc_matrix_auto-labels.tsv'
sector_matrix = pd.read_csv(sector_matrix_file, sep='\t')
sector_matrix = sector_matrix.set_index('naics')

eeeppat = pd.read_csv('/home/desktop/patstat_data/other_data/eee-ppat/EEE_PPAT_Oct2011.csv',
                    usecols = ['person_id', 'sector'])
eeeppat = eeeppat.set_index('person_id')

subs_file = '/home/desktop/patstat_data/all_code/dbClean/amadeus_subsidiaries/wrds_subsidiaries.txt'
subs = pd.read_csv(subs_file, sep='\t')
subs = subs.rename(columns = {'IDNR': 'company_id'})
subs['company_id'] = subs['company_id'].apply(lambda x: x.lower())
subs = pd.DataFrame(subs.groupby('company_id').size(), columns=['n_subsidiaries'])

common_names = pd.read_csv('/home/desktop/patstat_data/all_code/dbUtils/country_common_person_names.csv',sep='\t')
adding_type = 'whole'
while not adding_type in ['whole', 'label']:
    adding_type = str(raw_input("Whole data (whole) or just for labeling (label)? "))

# load psClassify output
psclass_all = '/home/desktop/patstat_data/all_code/psClassify/r_out/r_output_all.csv'
psclass_df = pd.read_csv(psclass_all, sep='\t')
psclass_varlist = ['patstat_id', 
                   'is_person', 
                   'certain_not_person', 
                   #'name',
                   #'country',
                   #'applnt_seq',
                   #'patent_ct',
                   #'applnt_score',
                   #'word_count',
                   #'avg_word_len',
                   #'string_len',
                   #'only_letters',
                   'lots_of_patents',
                   #'has_legal_out',
                   'has_legal_in',
                   'maybe_foreign_legal',
                   'has_first_name',
                   'pr_is_person',
                   'is_person_hat'
                   ]

psclass_df = psclass_df[psclass_varlist]
psclass_df = psclass_df.set_index('patstat_id')


def parallel_clear():
    dview.results.clear()
    par_client.results.clear()
    gc.collect()
    return

#cores = sys.argv[1]
cores = 10
#list_countries = sys.argv[2:]
list_countries = countries

"""
if not 'eu27' in list_countries:
    countries = list_countries
"""

for eu_country in countries:
    print 35*'-'
    print 'processing', eu_country.upper()
    print 35*'-'
    
    common_list = list([common_names.country == eu_country]['names'])[0].split()
    #print common_list
    
    # file names
    best_matches_file = input_bestmatches + 'best_matches_%s_wrds_v3.csv' % eu_country
    company_file = input_company + 'cleaned_output_%s.tsv' % eu_country
    patstat_file = input_patstat + 'patstat_geocoded_%s_v2.csv' % eu_country
    
    if adding_type == 'label':
        labeled_file = input_labels + '%s/' % eu_country + 'labeled_%s.csv' % eu_country       
        validation_file = input_labels + '%s/' % eu_country + 'validation_labeled_%s.csv' % eu_country       
        output_file = output_root + 'training_data_%s.tsv' % eu_country 
    else:
        output_file = output_root + 'whole/added_vars_whole_%s_wrds_v4.tsv' % eu_country 
         
    print 'reading company'
    company_db = pd.read_csv(company_file, sep='\t', header=0) #, dtype=typedict

    company_db.company_name = company_db.company_name.apply(as_str)
    company_db.company_id = company_db.company_id.apply(as_str)
    company_db = company_db.set_index('company_id')
    company_db = company_db.join(subs)
    company_db['n_subsidiaries'] = company_db['n_subsidiaries'].fillna(0)
    
    # patstat
    print 'reading patstat'
    patstat_db = pd.read_csv(patstat_file, sep='\t')

    patstat_db = patstat_db[['person_name',
                           'lat',
                           'lng',
                           'ipc_code',
                           'applicant_seq',
                           'patent_ct',
                           'person_id',
                           'name_abbreviated',
                           'year']]
                          
    patstat_db['pid1'] = patstat_db['person_id'].apply(lambda x: int(x.split('**')[0]))
    patstat_db = patstat_db.set_index('pid1').join(eeeppat).reset_index()
    patstat_db['person_class'] = patstat_db['sector'].fillna('UNKNOWN')
              
    patstat_db = patstat_db.set_index('person_id')
    
    # candidates
    print 'reading best matches'
    best_matches = pd.read_csv(best_matches_file, sep='\t', header=0)
    best_matches['patstat_id'] = best_matches['patstat_id'].apply(as_str)

    best_matches = best_matches[pd.notnull(best_matches['company_id'])]
    
    if adding_type == 'label':
        # labels: training set    
        labeled = pd.read_csv(labeled_file, sep='\t', header=None)
        if labeled.shape[1]>4:
            labeled.columns = ['patstat_id', 'patstat_name', 'company_id', 'company_name', 
                            'true_match', 'year', 'naics_descr']
        else:
            labeled.columns = ['patstat_id', 'company_id', 'true_match', 'year']

        labeled = labeled[['patstat_id', 'company_id', 'true_match', 'year']]
        labeled['sample'] = 'training'
        # labels: validation set
        val_labeled = pd.read_csv(validation_file, sep='\t', header=None)
        if val_labeled.shape[1]>4:
            val_labeled.columns = ['patstat_id', 'patstat_name', 'company_id', 'company_name', 
                            'true_match', 'year', 'naics_descr']
        else:
            val_labeled.columns = ['patstat_id', 'company_id', 'true_match', 'year']

        val_labeled = val_labeled[['patstat_id', 'company_id', 'true_match', 'year']]
        val_labeled['sample'] = 'validation'
        labeled = labeled.append(val_labeled)
        labeled = labeled.reset_index(drop=True)
        
        # id must be string
        labeled.patstat_id = labeled.patstat_id.apply(as_str)
        labeled.company_id = labeled.company_id.apply(as_str)

    dbsize = len(best_matches)

    amd_var_list = ['naics_2007',
                #'lat','lng',#'d_uo_company_id','g_uo_company_id',
                    'oprevenue', 'n_employees',
                    #'website',
                    'intangible_fa','bracket',
                    'n_subsidiaries']
        
    ps_var_list = [ 'ipc_code', 'year',
                    'name_abbreviated', 'patent_ct',
                    'person_class'
                    #'applicant_seqs'
                    ] 

    best_matches['patstat_name'] = best_matches['patstat_name'].apply(as_str)
    best_matches['patstat_id'] = best_matches['patstat_id'].apply(as_str)
    try:
        best_matches['ps_legal'] = best_matches['ps_legal'].fillna('').apply(as_str)
    except ValueError:
        best_matches['ps_legal'] = best_matches['ps_legal'].apply(as_str)
    best_matches['company_name'] = best_matches['company_name'].apply(as_str)
    try:
        best_matches['am_legal'] = best_matches['am_legal'].fillna('').apply(as_str)
    except ValueError:
        best_matches['am_legal'] = best_matches['am_legal'].apply(as_str)
    
    # join all the datasets
    print 'joining datasets...'
    #print labeled.iloc[1]
    
    def expand_pid(single_id):
        """
        From a single PATSTAT id, returns the group of PATSTAT ids 
        resulting from same-name merging.
        """
        for i, v in patstat_db.iterrows():
            if single_id in i:
                return i    

    if adding_type == 'label':
        unique_ids_df = pd.DataFrame(labeled.patstat_id.unique(), columns=['single_pid']) 
        unique_ids_df['patstat_id_list'] = unique_ids_df['single_pid'].apply(expand_pid)
        unique_ids_df = unique_ids_df.set_index('single_pid')
        labeled = labeled.set_index('patstat_id').join(unique_ids_df)
        labeled = labeled.reset_index(drop=True)
        labeled = labeled.set_index(['patstat_id_list', 'company_id'])
        labeled = labeled.join(best_matches.set_index(['patstat_id','company_id']))
        labeled.index.names = ['patstat_id', 'company_id']
        start_data = labeled.drop('year', 1).reset_index()
        start_data = start_data.dropna(0, subset=['lev_name_dist'])
    else:
        start_data = best_matches

    start_data = start_data.join(company_db[amd_var_list], on='company_id')
    larger_data = start_data.join(patstat_db[ps_var_list], on='patstat_id')

    larger_data = larger_data.set_index('patstat_id')
    larger_data.index.name = 'patstat_id'
    
    # make sure we have strings and not objects to avoid errors
    larger_data['patstat_name'] = larger_data['patstat_name'].apply(as_str)
    larger_data['company_name'] = larger_data['company_name'].apply(as_str)

    print bcolors.OKBLUE
    os.system('ipcluster start -n %s &' % str(cores))
    time.sleep(35)
    print bcolors.ENDC
    
    par_client = Client()
    dview = par_client[:]
    dview.block = True

    @dview.parallel(block=True)
    def calc_avg_f_am(name):
        """
        Returns the average frequency of the words in a name.
        Based on in-sample bag-of-words analysis.
        """
        return bagger_am.average_freq(name)
         
    #@dview.parallel(block=True)
    #def calc_avg_f_ps(name):
    #    return bagger_ps.average_freq(name)

    @dview.parallel(block=True)
    def calc_qavg_f_am(name):
        """
        Returns the average square frequency of the words in a name.
        Based on in-sample bag-of-words analysis.
        
        Compared to normal frequency, this should weight the more 
        common words more heavily.
        """
        return bagger_am.q_average_freq(name)
           
    #@dview.parallel(block=True)
    #def calc_qavg_f_ps(name):
    #    return bagger_ps.q_average_freq(name)

    @dview.parallel(block=True)
    def separate_legal(name):
        """
        Separates legal id from name, based on country.
        """
        return find_legal.separate_comp_legalid(name, eu_country)
            
    @dview.parallel(block=True)
    def dummy_perfect_match(name1, name2):
        """
        Dummy: 1 if the two names are exact matches.
        """
        return 0 if (1 - jellyfish.jaro_winkler(name1, name2))!=0 else 1
        
    @dview.parallel(block=True)
    def jw_parallel(name1, name2):
        """
        Calculates the Jaro-Winkler string distance between the two names.
        """
        return (1 - jellyfish.jaro_winkler(name1, name2))

    @dview.parallel(block=True)
    def lev_parallel(name1, name2):
        """
        Calculates the Levenshtein ratio string distance between the two names.
        """
        return (1 - Levenshtein.ratio(name1, name2))
            
    @dview.parallel(block=True)
    def drop_common_words(name):
        """
        Returns a name that does not include too frequent words (top50)
        """
        return bagger_both.remove_frequent(name)
        
    @dview.parallel(block=True)
    def metaphone_conv(name):
        """
        Returns the metaphone conversion of a name.
        """
        return dmeta_hash(''.join(name.split(' ')))[0]

    @dview.parallel(block=True)
    def common_first_name(name, commons = common_list):
        """
        Dummy: 1 if a common first name appears in the input name.
        """
        list_words = name.split()
        for w in list_words:
            if w in commons:
                return 1
        return 0
        
    @dview.parallel(block=True)
    def web_clean(name):
        """
        Cleans a name under the assumption that it is a website address.
        """
        if name=='nan' or len(name)==0:
                return ''
        try:
            name = name.replace('www','')
            name = name.replace('.',' ').strip()
            name = name.split()[0]
            return name
        except:
            return ''
        
    @dview.parallel(block=True)
    def geo_categorical(geodist):
        """
        Factor translation of the geographical distance. 
        Necessary to account for missing data.
        """
        if (not pandas.notnull(geodist)) or geodist=='':
            return 0
        elif geodist < 10:
                return 1
        elif geodist < 50:
                return 2
        elif geodist < 200:
                return 3
        else:
                return 4
                
    @dview.parallel(block=True)
    def min_jw_of_alt(patstatid):
        """
        For a PATSTAT id (grouped) returns a table in which every row
        corresponds to a company. 
        The return value for each company is the minimum Jaro-Winkler 
        distance between the PATSTAT name and the other companies.
        """
        subset_data = larger_data.loc[patstatid]
        min_dists = []
        try:
            if len(subset_data.company_id)>1:
                for am_id in subset_data.company_id:
                    min_dists.append(min(subset_data[subset_data.company_id != am_id].jw_name_dist))
            else:
                min_dists.extend([1]*len(subset_data.company_id))
            return min_dists
        except AttributeError:
            return [1]
        
    @dview.parallel(block=True)
    def max_sec_of_alt(patstatid):
        """
        For a PATSTAT id (grouped) returns a table in which every row
        corresponds to a company. 
        The return value for each company is the maximum sector 
        similarity between the PATSTAT name and the other companies.
        """
        subset_data = larger_data.loc[patstatid]
        max_sims = []
        try:
            if len(subset_data.company_id)>1:
                for am_id in subset_data.company_id:
                    max_sims.append(max(subset_data[subset_data.company_id != am_id].sector_sim_max))
            else:
                max_sims.extend([0]*len(subset_data.company_id))
            return max_sims
        except AttributeError:
            return [0]
        
    # Sync the necessary imports and path settings
    dview.execute('import sys')
    dview.execute('sys.path.append("/home/desktop/patstat_data/all_code/dbUtils/")')

    with dview.sync_imports():
        import bag_of_words as bofw
        import company_legal_id as legal
        import jellyfish
        import Levenshtein
        import fuzzy
        import pandas
        import numpy
            
    dview.execute('dmeta_hash = fuzzy.DMetaphone(10)')
    
    find_legal = legal.FindLegalId()
    dmeta_hash = fuzzy.DMetaphone(10)
    dview.push({'eu_country': eu_country})
    dview.push({'sector_matrix': sector_matrix})
    dview.push({'larger_data': larger_data})
    
    #dview.push({'common_list': common_list})

    start_time = time.time()
    
    # clean website
    #website = web_clean.map(list(larger_data['website'].apply(as_str).fillna(''))) 
    #print 'cleaned website'
    #larger_data['website'] = website
    #parallel_clear()
    
    # calculate jw of cleaned names
    jw_names = jw_parallel.map(list(larger_data['patstat_name']), list(larger_data['company_name'])) #
    print 'calculated jw of cleaned names'
    larger_data['clean_names_jw'] = jw_names
    parallel_clear()

    dview.push({'larger_data': larger_data})
    
    # calculate minimum jw of alternatives for same pid
    min_jw_alt = min_jw_of_alt.map(list(larger_data.index.unique())) #
    print 'calculated minimum jw of alternatives for same pid'
    larger_data['min_jw_of_alt'] = list(it.chain(*min_jw_alt))
    parallel_clear()  
    
    # calculate lev of cleaned names
    lev_names = lev_parallel.map(list(larger_data['patstat_name']), list(larger_data['company_name'])) #
    print 'calculated lev of cleaned names'
    larger_data['clean_names_lev'] = lev_names
    parallel_clear()

    # perfect match dummy
    perfect_matches = dummy_perfect_match.map(list(larger_data['patstat_name']), list(larger_data['company_name'])) #
    print 'calculated dummy for perfect matches'
    larger_data['perfect_match'] = perfect_matches
    parallel_clear()
    
    # calculate jw of legal
    jw_legal = jw_parallel.map(list(larger_data['ps_legal'].fillna('')), list(larger_data['am_legal'].fillna(''))) #
    print 'calculated jw of legal'
    larger_data['legal_jw'] = jw_legal
    parallel_clear()
        
    # sector distance
    maxd = []
    sumd = []
    for i, naics in enumerate(larger_data.naics_2007):
        freq = []
        ipcstr = larger_data.ipc_code.iloc[i]
        if isinstance(ipcstr, str):
            ipc_list = [c[:3] for c in ipcstr.split('**') if len(c)>1]
            for ipc in ipc_list:
                try:
                    ff = sector_matrix[ipc].loc[naics]
                except:
                    ff = 0.0
                freq.append(ff)
        else:
            freq.append(0.0)
        
        if len(freq) > 0:
            simsim = {'max': max(freq), 'sum': numpy.sum(freq)}
        else:
            simsim = {'max': 0, 'sum': 0}
            
        maxd.append(simsim['max'])
        sumd.append(simsim['sum'])
    
    larger_data['sector_sim_max'] = maxd
    larger_data['sector_sim_sum'] = sumd
    print 'calculated sector distance'
    
    dview.push({'larger_data': larger_data})
    
    # calculate maximum sector similarity of alternatives for same pid
    max_sec_alt = max_sec_of_alt.map(list(larger_data.index.unique())) #
    print 'calculated minimum sector dist of alternatives for same pid'
    larger_data['max_sec_of_alt'] = list(it.chain(*max_sec_alt))
    parallel_clear()  
    
    # calculate metaphone conversion of words in company_name
    metaphone_am = metaphone_conv.map(list(larger_data['company_name'])) 
    print 'metaphone conversion of company done'
    larger_data['metaphone_am'] = metaphone_am
    parallel_clear()  

        
    # calculate metaphone conversion of words in patstat_name
    metaphone_ps = metaphone_conv.map(list(larger_data['patstat_name'])) 
    print 'metaphone conversion of patstat done'
    larger_data['metaphone_ps'] = metaphone_ps
    parallel_clear()  
    
    larger_data['metaphone_am'].fillna('', inplace=True)
    larger_data['metaphone_ps'].fillna('', inplace=True)
    
    # calculate jw of metaphone conversions
    jw_metaphone = jw_parallel.map(list(larger_data['metaphone_ps']), list(larger_data['metaphone_am'])) #
    print 'calculated jw of metaphone conversions'
    larger_data['metaphone_jw'] = jw_metaphone
    parallel_clear()     

    # prepare for bag of words analysis          
    bagger_ps = bofw.BagOfWords(patstat_db.person_name.fillna(''))
    bagger_am = bofw.BagOfWords(company_db.company_name.fillna(''))
    bagger_both = bofw.BagOfWords(patstat_db.person_name.fillna('').append(company_db.company_name.fillna('')))

    dview.push({'bagger_am': bagger_am})
    dview.push({'bagger_ps': bagger_ps})
    dview.push({'bagger_both': bagger_both})          
                  
    # calculate average frequency of words in company_name
    avg_freq_am = calc_avg_f_am.map(list(larger_data['company_name'])) 
    print 'average frequencies (company) calculated'
    larger_data['avg_freq_am'] = avg_freq_am
    parallel_clear()

    # calculate average frequency of words in patstat_name
    #avg_freq_ps = calc_avg_f_ps.map(list(larger_data['patstat_name'])) 
    #print 'average frequencies (patstat) calculated'
    #larger_data['avg_freq_ps'] = avg_freq_ps
    #parallel_clear()

    # calculate average frequency of words in company_name
    qavg_freq_am = calc_qavg_f_am.map(list(larger_data['company_name'])) 
    print 'qaverage frequencies (company) calculated'
    larger_data['qavg_freq_am'] = qavg_freq_am
    parallel_clear()

    # calculate average frequency of words in patstat_name
    #qavg_freq_ps = calc_qavg_f_ps.map(list(larger_data['patstat_name'])) 
    #print 'qaverage frequencies (patstat) calculated'
    #larger_data['qavg_freq_ps'] = qavg_freq_ps
    #parallel_clear()
                
    # drop common words from patstat clean names
    dropped_common_ps = drop_common_words.map(list(larger_data['patstat_name'])) 
    print 'dropped too common words from patstat clean names'
    larger_data['name_less_common_ps'] = dropped_common_ps
    parallel_clear()
    
    # drop common words from company clean names
    dropped_common_am = drop_common_words.map(list(larger_data['company_name'])) 
    print 'dropped too common words from company clean names'
    larger_data['name_less_common_am'] = dropped_common_am
    parallel_clear()
    
    # categorize geodistance
    geo_cat = geo_categorical.map(list(larger_data['geo_dist'])) 
    print 'categorized geodistance'
    larger_data['geo_cat'] = geo_cat
    parallel_clear()
                 
    # calculate jw of patstat_name with company_web
    #jw_ps_web = jw_parallel.map(list(larger_data['patstat_name']), list(larger_data['website'])) #
    #print 'calculated jw of name v. web'
    #larger_data['ps_web_jw'] = jw_ps_web
    #parallel_clear()
      
    # calculate jw of decommonized names
    jw_inf_names = jw_parallel.map(list(larger_data['name_less_common_ps']), list(larger_data['name_less_common_am'])) #
    print 'calculated jw of cleaned names less common words'
    larger_data['name_less_common_jw'] = jw_inf_names
    parallel_clear()  
    
    # dummy for common first name in patstat name
    istherecommon = common_first_name.map(list(larger_data['patstat_name'])) #
    print 'calculated if there is a common first name'
    larger_data['common_first_name'] = list(istherecommon)
    parallel_clear() 
         
    print bcolors.OKBLUE
    os.system('ipcluster stop')
    time.sleep(5)              
    print bcolors.ENDC
    
    parallel_time = time.time() - start_time
    print eu_country.upper(), '- average time per record:', parallel_time/len(larger_data['patstat_name'])
    
    export_vars = ['company_id',                     #identifiers
                                'patstat_name','company_name',  #names w/o legal
                                'ps_legal','am_legal',          #legal
                                'name_less_common_ps',          #names w/o common
                                'name_less_common_am',
                                'metaphone_am',                 #metaphone conversion
                                'metaphone_ps',
                                #'website',                      #website clean
                                'perfect_match',                # perfect match dummy
                                'jw_name_dist','lev_name_dist', #string dist
                                'legal_jw',                     #legal string dist
                                'name_less_common_jw',          #uncommon name string dist           
                                'metaphone_jw',                 #metaphone string dist
                                #'ps_web_jw',                    #ps-web string dist
                                #'applicant_seqs',               #applicant score
                                'person_class',                 # person sector as in EEE-PPAT
                                'avg_freq_am',                  #average frequency
                                #'avg_freq_ps',
                                'qavg_freq_am',                 #q-average frequency
                                #'qavg_freq_ps',
                                'geo_dist',                     #geo distance
                                'geo_cat',
                                'year',                         #patstat info
                                'country',
                                'ipc_code',
                                'naics_2007',                   #company info
                                'sector_sim_max',
                                'sector_sim_sum',
                                'n_subsidiaries',
                                'n_employees',
                                'intangible_fa',
                                'oprevenue',
                                'bracket',
                                'min_jw_of_alt',                #min distance of alternatives
                                'max_sec_of_alt',
                                'common_first_name',            #is there a common first name in the patstatname?
                                'name_abbreviated',             #was an abbreviation used during pscleaning?
                                'patent_ct'
                                ]    
                                
    if adding_type == 'label':
        export_vars.extend(['true_match','sample'])         #learning vars    
            
    larger_data = larger_data[export_vars]
    
    larger_data = larger_data.join(psclass_df)
    
    larger_data.loc[pd.isnull(larger_data.pr_is_person), 'pr_is_person'] = 0
    larger_data['is_matchable'] = 1 - larger_data['pr_is_person']
    larger_data['is_not_person_hat'] = 1 - larger_data['is_person_hat']
    larger_data['is_not_person_hat'] = larger_data['is_not_person_hat'].fillna(1)
    larger_data.loc[larger_data.is_person == 1, 'is_matchable'] = 0
    
    larger_data.to_csv(output_file, sep='\t') 

