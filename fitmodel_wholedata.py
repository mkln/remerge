"""
This takes the results from the model fitting in R, and applies the results
to the whole data: for every grouping of patstat person_id's, this returns
the most likely matching company and the estimated probability of match.
"""

import pandas as pd
import numpy as np
import re
import os
from IPython.parallel import Client
import gc
import sys
import time

best_matches_folder = '/home/desktop/patstat_data/all_code/remerge/regression_data/whole/'
file_list = [file_name for file_name in os.listdir(best_matches_folder) if 'wrds' in file_name]

output_folder = '/home/desktop/patstat_data/all_code/remerge/matched_by_remerge/'

survived_vars = pd.read_csv('/home/desktop/patstat_data/all_code/remerge/R/guesses/survived_variables_wrds_v4.csv', sep='\t')

survived_vars = pd.Series(survived_vars.T[0]).drop('best_cutoff')

varnames = list(survived_vars.index)


countries = ['cz', "de", "fr", "gb", 
            "nl", 
            "it", "es", "be", "at", "dk", 
                "fi",
                'bg', 
        'sk', 'se', 'pt', 'pl', 'hu', 'ie', 'ee',
        'cy', 'si', 'ro',
        'lt', 'lu', 'lv', 'mt', 'gr'
        ]

#sel_countries = sys.argv[2:]
sel_countries = 'eu27'
if not sel_countries == 'eu27':
    countries = sel_countries

def colname_sub(col_name):
    for sep_this in factor_var_names:
        if sep_this in col_name:
            return sep_this, col_name.replace(sep_this, '')
    return col_name, ''
    
# load psClassify output
psclass_all = '/home/desktop/patstat_data/all_code/psClassify/r_out/r_output_all.csv'
psclass_df = pd.read_csv(psclass_all, sep='\t')
psclass_varlist = ['patstat_id', 
                   'is_person', 
                   'patent_ct',
                   'pr_is_person',
                   'is_person_hat'
                   ]

psclass_df = psclass_df[psclass_varlist]
psclass_df.pr_is_person[pd.isnull(psclass_df.pr_is_person)] = 0
psclass_df['is_matchable'] = 1 - psclass_df['pr_is_person']
psclass_df.is_matchable[psclass_df.is_person == 1] = 0
psclass_df = psclass_df[['patstat_id', 'patent_ct', 'is_matchable']].set_index('patstat_id')
   
# create 3 dicts: variables, factors, interactions
# they store the estimated coeffients
# 'interactions' is a dict of dict. usage: interactions['country', 'naics_2007']['si', '3359'] = 3.4296...
# use them in function that creates p_hat and y_hat for every record
# also create a list of variables to select relevant keys from row

for country in countries:

    file_name = 'added_vars_whole_%s_wrds_v4.tsv' % country

    print 'reading %s ...' % country
    country_bests = pd.read_csv(best_matches_folder + file_name, sep='\t')
    print 'shape of initial file ', country_bests.shape
    country_bests = country_bests.set_index('patstat_id').join(psclass_df.drop(['patent_ct', 'is_matchable'],1))
    country_bests = country_bests.reset_index().set_index(['patstat_id','company_id'])
    input_df = country_bests
    
    input_df.n_employees = input_df.n_employees.fillna(52) # mean in the training set
    input_df.n_subsidiaries = input_df.n_subsidiaries.fillna(0) # mean in the training set
    input_df.oprevenue = input_df.oprevenue.fillna(0) # mean in the training set
    input_df.intangible_fa = input_df.intangible_fa.fillna(0)
    input_df.patent_ct = input_df.patent_ct.fillna(1)
    input_df.geo_cat = input_df.geo_cat.fillna(-1)
    input_df.bracket = input_df.bracket.fillna(-1)
    input_df.geo_dist = input_df.geo_dist.fillna(260) # mean in the training set
    input_df.is_matchable = input_df.is_matchable.fillna(1)
    
    input_df['int_patents'] = (input_df['intangible_fa'] * input_df['patent_ct']).fillna(0)
    input_df['jw2_name'] = input_df['jw_name_dist'] **2
    input_df['lev2_name'] = input_df['lev_name_dist'] **2
    input_df['geo_cat1'] = ((input_df['geo_cat'] == 1).astype('int64')).fillna(0)
    input_df['naics_20073273'] = ((input_df['naics_2007'] == 3273).astype('int64')).fillna(0)
    input_df['naics_20073324'] = ((input_df['naics_2007'] == 3324).astype('int64')).fillna(0)
    input_df['naics_20073359'] = ((input_df['naics_2007'] == 3359).astype('int64')).fillna(0)
    input_df['naics_20073331'] = ((input_df['naics_2007'] == 3331).astype('int64')).fillna(0)
    input_df['bracket3'] = ((input_df['bracket'] == 3).astype('int64')).fillna(0)

    for vn in varnames:
        if not vn in input_df.columns:
            print 'adding %s' % vn
            # factor - factor interactions
            if 'interaction' in vn:
                cleaned = vn.replace(')','-').replace('.',', ').split('(')[1]
                variables = tuple(cleaned.split('-')[0].split(', '))
                type_of_var = tuple([input_df[variables[0]].dtype, input_df[variables[1]].dtype])
                values = tuple(cleaned.split('-')[1].split(', '))
                first_var_col = (input_df[variables[0]] == np.array(values[0]).astype(type_of_var[0]).item(0))
                second_var_col = (input_df[variables[1]] == np.array(values[1]).astype(type_of_var[1]).item(0))
                input_df[vn] = first_var_col * second_var_col
            elif ':' in vn:
                # factor - numeric interactions
                if 'person_classUNKNOWN' in vn:
                    other_variable = vn.split(':')[1]
                    input_df[vn] = (input_df['person_class'] == 'UNKNOWN') * input_df[other_variable].fillna(0)
                else:
                    first_variable = vn.split(':')[0]
                    second_variable = vn.split(':')[1]
                    input_df[vn] = input_df[first_variable] * input_df[second_variable]
            input_df[vn] = input_df[vn].fillna(0)
                
    dotted = input_df[varnames].dot(survived_vars[varnames])
    probs = 1.0/(1+dotted.apply(lambda x: np.exp(-x)))
    
    results = pd.DataFrame(probs, columns = ['phat'])
    results = results.reset_index()
    results = results.sort(['patstat_id','phat'], ascending = [True, False])
    results = results.groupby(['patstat_id']).agg({
                        'company_id' : lambda x: x.head(1),
                        'phat': lambda x: x.head(1)})
    
    results.to_csv(output_folder + 'wrds_remergeed_%s_v5.csv' % country, sep='\t')
    
    print 'done with %s' % country

