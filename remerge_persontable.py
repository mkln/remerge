"""
Takes the matching results and returns a table of
patstat_id(ungrouped) : amadeus_id : probability of match

The resulting table can then be loaded into an SQL server.
"""

import pandas as pd
from IPython.parallel import Client
import time
import gc
import os

eu27 = ['at', 'bg', 'be', 'it', 
        'gb', 
        'fr', 'de',
        'sk', 'se', 'pt', 'pl', 'hu', 'ie', 'ee',
        'es', 'cy', 'cz', 
        'nl',
        'si', 'ro', 'dk',
        'lt', 'lu', 'lv', 'mt', 'fi', 'gr']
        
remerge_folder = '/home/desktop/patstat_data/all_code/remerge/matched_by_remerge/'

remerge = pd.DataFrame()
remerge_filelist = [remerge_folder + 'wrds_remergeed_%s.csv' % country for country in eu27]

remerge = []
for country in eu27:
    remerge_f = remerge_folder + 'wrds_remergeed_%s_v4.csv' % country
    print remerge_f
    tempdf = pd.read_csv(remerge_f, sep='\t')
    print tempdf.shape
    tempdf = tempdf[['patstat_id', 'company_id', 'phat']]
    #tempdf['country'] = country
    remerge.append(tempdf)

remerge = pd.concat(remerge)
remerge = remerge.reset_index(drop=True)

def ungroupby(df, colname, sep):
    begin_time = time.time()
    
    s = df[colname].apply(lambda x: pd.Series(x.split(sep)))
    
    run_time = time.time() - begin_time
    print 'single: ran in %s seconds' % str(run_time)
    
    s = s.stack()
    s.index = s.index.droplevel(-1) 
    s.name = colname
    return_df = df.drop(colname, 1)
    return_df = return_df.join(s)
    return return_df



matchy = ungroupby(remerge, 'patstat_id', sep='**')

matchy.to_csv('/home/desktop/patstat_data/all_code/remerge/remerge_sqllike_table/remerge_patstatOct2011_AmadeusWRDS.csv', sep='\t', index = False)










