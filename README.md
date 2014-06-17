starting point:
- dbClean for raw output

- remerge_filter.py
for every patstat name, computes JW and Lev string distances, 
then for every patstat name outputs Union(top10lw, top10lev),
includes computation of geo-location
includes separation of names and legal identifiers
adds amadeus variables for hand labeling later.
this is the most resource-intensive part of the algorithm

- remerge_sample.py
loads RAW patstat and amadeus
loads candidate matches
takes previous dataset and asks user to find the true matches

- build_sector_matrix_with_automatch.py


- remerge_genvars.py
for every row in the previously output dataset, imports/creates new variables:
	import ipc code list from patstat
	import naics
	import n_employees
	import revenue bracket
	import n_subsidiaries
	import intangibles
	create average frequency of ps words
	create average frequency of am words
	create average q frequency of ps words
	create average q frequency of am words
	create jw distance after removal of common words
	create jw distance of legal ids
	
- remerge_pre_modelfit.py
	creates dummy from naics
	creates dummy from ipc

	*calls R for estimation

- remerge_persontable.py
- remerge_fitmodel_training.r
- remerge_fitmodel_wholedata.py

