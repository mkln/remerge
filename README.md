# REMERGE #
## Linking PATSTAT to Company databases ##

### Introduction ###
PATSTAT is a database published by the European Patent Office and includes info on millions of patents and patentees.
Its usage is sometimes limited by the little information on the patentees.
Linking it to company databases has historically been a manual task. This is due to:
- its focus on the patent applications, not on the patent applicants or inventors,
- missing classification of patentees into categories such as individuals, companies, other organizations,
- missing basic information on patentees such as their address, or their name.
In addition, large company databases include a large majority of non-patenting companies.
Ultimately, just a few patentees should be matched to a relatively small number of companies.
For these reasons, advanced matching algorithm have not been used, as they make comparisons using the shared fields.

Remerge is a set of python scripts that allows to match PATSTAT to Company databases (in this case, Amadeus from Bureau van Dijk).
It is not limited to comparisons between shared fields, and uses as much information as possible. 
A Lasso-regression model is estimated on the training set and applied to the data to get the estimated probabilities of matching.

### Procedure ###
Starting from cleaned and geocoded data:

1.  __filter_companies.py__
For every PATSTAT name, computes JW and Lev string distances, 
then for every PATSTAT name outputs Union(top10lw, top10lev),
includes computation of geo-location
includes separation of names and legal identifiers
adds Amadeus variables for hand labeling later.
this is the most resource-intensive part of the algorithm

2.  __extract_sample.py__
loads RAW PATSTAT and Amadeus, loads candidate matches,
takes previous dataset and asks user to find the true matches

3.  __remerge_sector_matrix.py__
calculates IPC-NAICS "similarity" by looking into the unique exact matches.
A unique exact match is, of all pairings between a PATSTAT name and a company,
the only one in which the two names are the same. 
Most PATSTAT names have no exact match. Unique ones are even less.

4.  __generate_vars.py__
5.  __prepare_modelfit.py__
Generate some of the variables that are used by the Lasso-regression.

6.  __remerge_fitmodel_training.r__
Fits the Lasso-regression model. (Calls some python code)
Loads R source code from __regression_functions-modelmatrix.r__

7.  __remerge_fitmodel_wholedata.py__
Fits the generated model to the whole dataset. Saves the results.

8.  __remerge_persontable.py__
Takes the matching results and returns a table of
_patstat_id_ : _phat_ : _company_id_ 
where _patstat_id_ is the same as _person_id_ in patstat and _phat_ is the estimated probability of match.
The resulting table can then be loaded into an SQL server.
