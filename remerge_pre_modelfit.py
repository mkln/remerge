"""
Before regression is run, creates dummy variables for grouped values.
These are patent years (have been grouped together if same perso_id
is associated to different patents in different years), and IPC codes
(as multiple IPC codes are usually associated to a single patent, and 
more than one patent can be associated to a single person_id).

Returns the dataset to be read by R for model fitting.
"""

import pandas as pd
import os
import re
import glob


input_root = '/home/desktop/patstat_data/all_code/remerge/regression_data/whole/'
output_root = '/home/desktop/patstat_data/all_code/remerge/R/inputs/'

# load training set
training = pd.read_csv('/home/desktop/patstat_data/all_code/remerge/labeled_sample/labeled_all.csv',
                        sep = '\t', header = None)
training.columns = ['patstat_id', 'company_id', 'true_match']

# load addedvars files
avdf = []
for f in [f for f in glob.glob(input_root + '*v3.tsv')]:
    print 'processing %s...' % f
    temp = pd.read_csv(f, sep='\t')
    avdf.append(temp)
av = pd.concat(avdf)

tr_av = training.set_index(['patstat_id','company_id'])
tr_av = tr_av.join(av.set_index(['patstat_id','company_id']), how='inner')
#tr_av = av.set_index(['patstat_id','company_id'])
tr_av.index.names = ['patstat_id','company_id']
tr_av = tr_av.reset_index()

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
psclass_df.loc[pd.isnull(psclass_df.pr_is_person), 'pr_is_person'] = 0
psclass_df['is_matchable'] = 1 - psclass_df['pr_is_person']
psclass_df['is_not_person_hat'] = 1 - psclass_df['is_person_hat']
psclass_df['is_not_person_hat'] = psclass_df['is_not_person_hat'].fillna(1)
psclass_df.loc[psclass_df.is_person == 1, 'is_matchable'] = 0


df = tr_av.set_index('patstat_id').join(psclass_df.set_index('patstat_id'))

print 'starting with', df.shape

var_list = [#u'patstat_id',
                #u'pid',
                u'company_id',
                u'patstat_name',u'ps_legal',
                u'company_name',u'am_legal',
                #u'sample',
                u'true_match',
                u'perfect_match',
                u'jw_name_dist',
                u'lev_name_dist',
                u'legal_jw',
                u'name_less_common_jw',
                u'metaphone_jw',
                #u'ps_web_jw',
                #u'applicant_seqs',
                u'person_class',
                u'avg_freq_am',
                #u'avg_freq_ps',
                u'qavg_freq_am',
                #u'qavg_freq_ps',
                u'geo_dist',
                u'geo_cat',
                u'year',
                u'country',
                u'ipc_code',
                u'naics_2007',
                u'sector_sim_max',
                u'sector_sim_sum',
                u'n_subsidiaries',
                u'n_employees',
                u'intangible_fa',
                u'bracket',
                u'min_jw_of_alt',
                u'max_sec_of_alt',
                u'patent_ct',
                u'lots_of_patents',
                u'has_legal_in',
                u'maybe_foreign_legal',
                u'is_matchable',
                u'is_not_person_hat',
                u'has_first_name',
                u'name_abbreviated',
                u'oprevenue']
                
#country_df = country_df.reset_index()
regress_df = df[var_list]
regress_df.loc[pd.isnull(regress_df.is_matchable), 'is_matchable'] = 1

# prepare vars for regression

def squeeze(char,s, trunc = 3):
    while char*2 in s:
        s=s.replace(char*2,char)
    return ' '.join([s[:trunc] for s in s.split(char)]).strip()


# year and ipc codes
regress_df['year'] = regress_df['year'].fillna('year_missing')
regress_df['ipc_code'] = regress_df['ipc_code'].fillna('ipc_missing')

ipc_df = regress_df['ipc_code'].reset_index().drop_duplicates().set_index('patstat_id')
year_df = regress_df['year'].reset_index().drop_duplicates().set_index('patstat_id')

fixed_ipc = []
for i,v in ipc_df.iterrows():
    fixed_ipc.append(squeeze('*', v.values[0]) if v.values[0] != 'ipc_missing' else 'ipc_missing')
    
fixed_year = []
for i,v in year_df.iterrows():
    fixed_year.append(squeeze('*', v.values[0], 4) if v.values[0] != 'year_missing' else 'year_missing')
   
iatom_col = pd.Series([xx for xx in ' '.join(fixed_ipc).replace('  ',' ').split(' ') if len(xx)>2]).unique()
yatom_col = pd.Series([xx for xx in ' '.join(fixed_year).replace('  ',' ').split(' ') if len(xx)>2]).unique()

# fixed because of memory errors
# potentially can run in parallel
for x, col in enumerate(yatom_col):
    if not x % 10:
        print '%d' % (100.0*x/len(yatom_col))
    year_df[col] = year_df['year'].apply(lambda x: int(col in x))
    
for x, col in enumerate(iatom_col):
    if not x % 10:
        print '%d' % (100.0*x/len(iatom_col))
    ipc_df[col] = ipc_df['ipc_code'].apply(lambda x: int(col in x))



regress_df = regress_df.drop(['year','ipc_code'], 1)
regress_df = regress_df.join(year_df.drop('year', 1)).join(ipc_df.drop('ipc_code', 1))

print 'dropped dummy-generating vars:', regress_df.shape, '(final)'

output_file = 'wrds_v2_all.csv'
regress_df.index.name = 'patstat_id'
regress_df.to_csv(output_root + output_file, sep='\t')

