"""
Extracts a sample of PATSTAT names from the best matches files 
and asks the user to identify if there is a match, and if so, 
which company is the true match. 

Note: the sample will be divided into two for model fitting.
Expect to find matching companies around 30-40% of the times.

Author: Michele Peruzzi
"""

# should be changed to include the data used for postprocessing
# saves time later.

import pandas as pd
import os
import glob
import numpy as np
import time

import sys
sys.path.append("/home/desktop/patstat_data/all_code/dbUtils/")
from ccolors import *


# i/o folders
best_matches_root = '/home/desktop/patstat_data/all_code/remerge/best_matches/'
patstat_clean_root = '/home/desktop/patstat_data/all_code/dbClean/patstat_geocoded/'
amadeus_clean_root = '/home/desktop/patstat_data/all_code/remerge/amadeus_wrds/'
titles_root = '/home/desktop/patstat_data/all_code/remerge/ps_titles/'
labeled_root = '/home/desktop/patstat_data/all_code/remerge/labeled_sample/'

# record loading times
begin_time = time.time()

print 'loading best matches...'
best_matches = [ pd.read_csv(f, sep='\t') for f in glob.glob(best_matches_root + '*wrds_v3.csv')
                    ]
best_matches = pd.concat(best_matches)

print 'loading patstat (cleaned and geocoded)...'
patstat_clean = [ pd.read_csv(f, sep='\t') for f in glob.glob(patstat_clean_root + '*_v2.csv')
                    ]
patstat_clean = pd.concat(patstat_clean)
patstat_clean = patstat_clean.set_index('person_id')

print 'extract random sample of consolidated patstat cleaned names...'
# create df with it
sample_id = np.random.choice(best_matches.patstat_id.unique(), 100)

sample_df = pd.DataFrame(sample_id, columns = ['patstat_id'])
sample_df = sample_df.set_index('patstat_id')
sample_df = sample_df.join(patstat_clean)
sample_df.index.name = 'patstat_id'
sample_df = sample_df.reset_index()
sample_df['pid_1'] = sample_df['patstat_id'].apply(lambda x: x.split('**')[0])

# no need for patstat in memory anymore
del patstat_clean

print 'reverse geocode the coordinates (faster)...'
locations = pd.read_csv('/home/desktop/patstat_data/all_code/fuzzygeo/latlong_dict.csv')
locations = locations.groupby(['Lat','Lng']).agg({'City': lambda x: '**'.join(x.values)})
locations = locations.rename(columns = {'City':'patstat_city'})

print 'add city to sample dataframe...'
sample_df = sample_df.set_index(['lat','lng'])
sample_df = sample_df.join(locations)
sample_df.index.names = ['lat','lng']
sample_df = sample_df.reset_index()
sample_df = pd.merge(sample_df, best_matches, how='left')
del locations

print 'finally load amadeus and merge raw data to frame...'
amadeus_clean = [ pd.read_csv(f, sep='\t') for f in glob.glob(amadeus_clean_root + '*.tsv')]
amadeus_clean.extend([ pd.read_csv(f, sep='\t') for f in glob.glob(amadeus_clean_root + '*.txt' )])

amadeus_clean = pd.concat(amadeus_clean)
amadeus_clean = amadeus_clean.drop(['lat','lng', 'year', 'country', 'company_name'], 1)
amadeus_clean = amadeus_clean.rename(columns = {'city':'amadeus_city'})

print 'and join with sample dataframe...'
sample_df = pd.merge(sample_df, amadeus_clean, 
                how='left', 
                left_on='company_id', 
                right_on='company_id')
del amadeus_clean

print 'join naics descriptions...'
naics_file = '/home/desktop/patstat_data/all_code/dbUtils/naics_descriptions.csv'
naics_df = pd.read_csv(naics_file, sep='\t')
sample_df = pd.merge(sample_df, naics_df, how='left', on='naics_2007')

print 'join titles...'
combi_list = []
for i, row in sample_df.iterrows():
    combi_list.extend([y + '_' + row['country'] + '.csv'
                            for y in row['year'].split('**')])
combi_list = list(set(combi_list))
titles = [ f for f in [titles_root + 'title_extract_' + c 
                                    for c in combi_list
                         ]
         ]
titles_df = []
for title in titles:
    try:
        temp = pd.read_csv(title, sep='\t')
        titles_df.append(temp)
    except IOError:
        continue
        
titles_df = pd.concat(titles_df)
titles_df = titles_df.rename(columns={'patstat_id':'pid_1'})
titles_df = titles_df[['pid_1', 'titles']]
titles_df['pid_1'] = titles_df['pid_1'].astype('S') 
titles_df = titles_df.drop_duplicates()
titles_df = titles_df.groupby('pid_1'
                        ).agg({'titles': lambda x: ' '.join(x.values)}
                        ).reset_index()
sample_df = pd.merge(sample_df, titles_df, 
                     how='left',
                     on = 'pid_1')
del titles_df

sample_df = sample_df.set_index('patstat_id')
sample_df = sample_df[pd.notnull(sample_df.company_name)]

try:
    already_matches = pd.read_csv(labeled_root + 'labeled_all_v2.csv', 
                                sep='\t', header=None)
    #last_pid = already_matches[0].tail(1).values[0]
    already = len(already_matches[0].unique())
    found_matches = already_matches[2].sum()
except:
    already = 0
    found_matches = 0

end_time = time.time()

sample_df = sample_df.drop_duplicates()

pd.set_option('display.width', 1000)
for i, pid in enumerate(sample_id):
    os.system('clear')
    if i==0:
        print end_time - begin_time, 'seconds to load all the data.'
    pselect = sample_df.loc[pid].reset_index()[['person_name',
                                    'year',
                                    'patstat_city', 
                                    'country',
                                    'patent_ct',
                                    #'titles'
                                    ]].head(1)
    titlestr = sample_df.loc[pid].reset_index()['titles'].values[0]
        
    print bcolors.OKBLUE + 'STATUS:'
    print '%s done (%s total)' % (str(i), str(already+i))
    print '%s found' % str(found_matches)
    print bcolors.BOLD + bcolors.OKGREEN
    print 120*'-'
    print pselect.to_string()
    print titlestr
    print 120*'-'
    print bcolors.ENDC
        
    displaydf = sample_df.loc[pid][['company_name',
                                  'jw_name_dist',
                                  'amadeus_city', 
                                  'naics_2007_d',
                                  'bracket',
                                  'n_employees',
                                  'intangible_fa',
                                  'oprevenue',
                                  'research',
                                  'company_id'
                                  ]]
    displaydf = displaydf.sort('jw_name_dist').reset_index()

    print displaydf.drop('patstat_id',1).to_string()

    print bcolors.BOLD + bcolors.WARNING
    print 'If there is a match, type its number, otherwise press <Enter> '
    g = raw_input('Type "none" if there is a match, but can\'t say which one. ')
    print bcolors.ENDC
        
    if g == 'exit':
        break
            
    if g!= 'none':
        try:
            displaydf['true_match'] = 0
            displaydf['true_match'].iloc[int(g)] = 1
            print displaydf[['patstat_id', 'company_id', 'true_match']].to_string()
            true_match_df = (displaydf[['patstat_id','company_id','true_match']])       
            time.sleep(0.5)
            found_matches += 1
            true_match_df.to_csv(labeled_root + 'labeled_all_r3.csv', 
                            sep='\t', index=False, mode='a', header=False)
        except:
            displaydf['true_match'] = 0
            print displaydf[['patstat_id', 'company_id', 'true_match']].to_string()
            true_match_df = (displaydf[['patstat_id', 'company_id',  'true_match']])       
            time.sleep(0.5)
            true_match_df.to_csv(labeled_root + 'labeled_all_r3.csv', 
                            sep='\t', index=False, mode='a', header=False)
            
    

