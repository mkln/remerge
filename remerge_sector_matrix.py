"""
Creates a table with the similarity between patent IPC codes and
company NAICS2007 sector codes. 
This is based on unique exact matches between PATSTAT names and company names.
Assumes that unique exact matches are true matches.

Takes the best matches lists, the cleaned PATSTAT and cleaned Amadeus files
as inputs, and returns a tab-separated file with the sector similarity measures.

"""

import pandas as pd
import os
import re

root_folder = '/home/desktop/patstat_data/all_code/remerge/labeled_sample/'
amadeus_folder = '/home/desktop/patstat_data/all_code/remerge/amadeus_wrds/'
patstat_folder = '/home/desktop/patstat_data/all_code/remerge/patstat_input/'
best_matches_root = '/home/desktop/patstat_data/all_code/remerge/best_matches/'

eu27 = ['at', 'bg', 'be', 'it', 'gb', 'fr', 'de', 'sk', 'se', 'pt', 'pl',
        'hu', 'ie', 'ee', 'es', 'cy', 'cz', 'nl', 'si', 'ro', 'dk', 'lt',
        'lu', 'lv', 'mt', 'fi', 'gr']
    

"""
file_list = []
folder_list = [fo for fo in os.listdir(root_folder) if fo[-3:]!='tsv']
for folder in folder_list:
    all_labels = os.listdir(root_folder + folder)
    all_labels = [(folder + '/' + f) for f in all_labels if f[:3] == 'lab']
    file_list.extend(all_labels)
"""

labels_df = pd.DataFrame()
for country in eu27:
    best_matches = best_matches_root + 'best_matches_%s_wrds.csv' % country
    matches_df = pd.read_csv(best_matches, sep=',')
    matches_df.sort(columns=['patstat_name','jw_name_dist'], inplace=True)
    matches_df = matches_df[['patstat_name','company_name','patstat_id','company_id','jw_name_dist']]
    matches_df['patstat_id_list'] = matches_df['patstat_id'] 
    pid = []
    for i,v in matches_df.iterrows():
        identif = re.sub('[*]+','*',str(v['patstat_id'])).split('*')
        if isinstance(identif, list):
            if identif[0]=='':
                identif = identif[1]
            else:
                identif = identif[0]
        pid.append(identif)
    matches_df['patstat_id'] = pid
        
    matches_df['patstat_id'] = matches_df['patstat_id'].astype(str)
    matches_df['company_id'] = matches_df['company_id'].astype(str)
    matches_df['true_match'] = 0
    matches_df['true_match'][matches_df.jw_name_dist == 0] = 1
    matches_df = matches_df[matches_df['true_match'] == 1][['patstat_id', 'company_id']]

    pid_unique_list = []
    for nn, pid in enumerate(matches_df.patstat_id.unique()):
        if not nn%1000:
            print country, 100*float(nn) / len(matches_df.patstat_id.unique()), 'percent'
        nmatches = len(matches_df[matches_df.patstat_id == pid])
        if nmatches==1:
            pid_unique_list.append(pid)
    
    labels_df = labels_df.append(matches_df[matches_df.patstat_id.isin(pid_unique_list)])

    labels_df['patstat_id'] = labels_df['patstat_id'].astype(str)
    labels_df['company_id'] = labels_df['company_id'].astype(str)


country_list = []
prev_country = ''
naics = []
for i, company_id in enumerate(labels_df.company_id):
    if not i%1000:
        print 100*float(i) / len(labels_df.company_id), 'percent'
    country = company_id[:2]
    country_list.append(country)
    amadeus_file_name = 'cleaned_output_%s.tsv' % country
    if country != prev_country:
        am_df = pd.read_csv(amadeus_folder + amadeus_file_name, sep='\t')
        #am_df.set_index('bvdep_id')
    naics.append(am_df[am_df['company_id'] == company_id].naics_2007.values[0])
    prev_country = country

naics = pd.Series(naics)

ipc = []
prev_country = ''
for i, patstat_id in enumerate(labels_df.patstat_id):
    if not i%1000:
        print 100*float(i) / len(labels_df.patstat_id), 'percent'
    country = country_list[i]
    patstat_file_name = 'patstat_geocoded_%s.csv' % country
    if country != prev_country:
        ps_df = pd.read_csv(patstat_folder + patstat_file_name, sep='\t')
        ps_df.person_id = ps_df.person_id.astype(str)
    appendy = ps_df[ps_df.person_id == patstat_id].ipc_code.values
    try:
        ipc.append(appendy[0])
    except:
        ipc.append('')
    prev_country = country
    
ipc = pd.Series(ipc)

mtr = pd.DataFrame(naics, columns = ['naics'])
mtr['ipc'] = ipc

mtr = mtr.dropna()
mtr = mtr.reset_index()
mtr = mtr.drop('index', 1)

# ipc dummies
for i,v in enumerate(mtr['ipc'][:]):
    mtr['ipc'][i] = re.sub('[*]+',' ',v)
   
ipc3 = []
for code in mtr.ipc:
    ipc3.append(' '.join([c[:3] for c in code.split()]))
mtr['ipc3'] = ipc3

dummies = pd.get_dummies(mtr['ipc3'])    
atom_col = pd.Series(' '.join(mtr['ipc3']).replace('  ',' ').split(' ')).unique()

#atom_col = [c for c in dummies.columns if '*' not in c]

for col in atom_col:
    mtr[col] = dummies[[c for c in dummies.columns if col in c]].sum(axis=1)
    #print dummies[[c for c in dummies.columns if col in c]].sum(axis=1)

mtr = mtr.drop(['ipc', 'ipc3'], 1)

matrix_summed = mtr.groupby('naics').sum()

for i, c in enumerate(matrix_summed):
    matrix_summed[c] =  matrix_summed[c].astype(float) / matrix_summed.sum(1)

matrix_summed.to_csv(root_folder + 'naics_ipc_matrix_auto-labels.tsv', sep='\t')

