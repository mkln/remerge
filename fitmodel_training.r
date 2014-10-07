# create dummies for training data and run lasso
# then create dummies for country data and apply model?

library(rPython)
# let python do the hard work with data... 
python.exec("
import pandas as pd
from glob import glob

in_folder = '/home/desktop/patstat_data/all_code/remerge/regression_data/whole/'
df_list = []
for f in glob(in_folder + '*v4.tsv'):
    print 'reading %s' % f
    df_list.append(pd.read_csv(f, sep='\t'))
df = pd.concat(df_list)

labels = pd.read_csv('/home/desktop/patstat_data/all_code/remerge/labeled_sample/labeled_all.csv',
                    sep='\t', header=None)
labels.columns = ['patstat_id','company_id','true_match']
labels = labels.set_index(['patstat_id','company_id'])
labels = labels.join(df.set_index(['patstat_id','company_id']))

print 'missing: ', pd.isnull(labels.perfect_match).sum()

labels = labels[pd.notnull(labels.perfect_match)]

labels.to_csv(in_folder + 'rPython_merged_data.csv', sep='\t')
")


library(glmnet)
library(dplyr)

library(ROCR)

setwd('/home/desktop/patstat_data/all_code/remerge/regression_data/whole/')
source("/home/desktop/patstat_data/all_code/remerge/regression_functions-modelmatrix.r")

lab_withvars <- read.table('rPython_merged_data.csv', header=TRUE, sep='\t')
#lab_withvars <- lab_withvars[complete.cases(lab_withvars$jw_name_dist),]
lab_withvars <- multi_dummy(lab_withvars, "patstat_id", "ipc_code", 3)
lab_withvars <- multi_dummy(lab_withvars, "patstat_id", "year", 4)

lab_withvars$naics_2007[is.na(lab_withvars$naics_2007)] <- "missing"
lab_withvars$oprevenue[is.na(lab_withvars$oprevenue)] <- 0
lab_withvars[,"naics_2007"] <- as.factor(lab_withvars[,"naics_2007"])
lab_withvars[,"bracket"] <- as.factor(lab_withvars[,"bracket"])
lab_withvars[,"geo_cat"] <- as.factor(lab_withvars[,"geo_cat"])

# var names
ids <- c("patstat_id","company_id","patstat_name","ps_legal","company_name","am_legal")

true_match <- "true_match"
vars <- c("jw_name_dist","lev_name_dist","legal_jw",
            "name_less_common_jw","metaphone_jw", "person_class",#"ps_web_jw", 
            #"applicant_seqs",
            "avg_freq_am",
            #"avg_freq_ps",
            "qavg_freq_am",
            #"qavg_freq_ps",
            "geo_dist", "country",
            "oprevenue",
            "sector_sim_max","sector_sim_sum","n_subsidiaries",
            "n_employees", "intangible_fa", "min_jw_of_alt", "max_sec_of_alt",
             "has_first_name", 
            "naics_2007", "bracket", "geo_cat", 
            "is_matchable", "lots_of_patents",
            "has_legal_in", "maybe_foreign_legal",
            "patent_ct", "perfect_match", "name_abbreviated")
    
year_names <- sapply(1990:2011, function(x) paste0("X",as.character(x)))
#start_ipc <- 1 + which(colnames(lab_withvars) == "oprevenue")
#end_ipc <- min(which(colnames(lab_withvars) %in% year_names)) - 1
ipc_names <- c(colnames(lab_withvars)[nchar(colnames(lab_withvars)) == 3], "ipc_missing")

#handling of vmissing data
#find variables with vmissing data

countna(lab_withvars)

lab_withvars$lev_name_dist <- as.numeric(as.character(lab_withvars$lev_name_dist))
lab_withvars$jw_name_dist <- as.numeric(as.character(lab_withvars$jw_name_dist))
lab_withvars$metaphone_jw <- as.numeric(as.character(lab_withvars$metaphone_jw))
lab_withvars$avg_freq_am <- as.numeric(as.character(lab_withvars$avg_freq_am))
lab_withvars$qavg_freq_am <- as.numeric(as.character(lab_withvars$qavg_freq_am))
lab_withvars$geo_dist <- as.numeric(as.character(lab_withvars$geo_dist))
lab_withvars$sector_sim_max <- as.numeric(as.character(lab_withvars$sector_sim_max))
lab_withvars$sector_sim_sum <- as.numeric(as.character(lab_withvars$sector_sim_sum))
lab_withvars$min_jw_of_alt <- as.numeric(as.character(lab_withvars$min_jw_of_alt))
lab_withvars$max_sec_of_alt <- as.numeric(as.character(lab_withvars$max_sec_of_alt))
lab_withvars$pr_is_person <- as.numeric(as.character(lab_withvars$pr_is_person))
lab_withvars$is_matchable <- as.numeric(as.character(lab_withvars$is_matchable))
lab_withvars$name_less_common_jw <- as.numeric(as.character(lab_withvars$name_less_common_jw))
lab_withvars$legal_jw <- as.numeric(as.character(lab_withvars$legal_jw))

lab_withvars <- subna(lab_withvars, "geo_dist", mean(lab_withvars$geo_dist, na.rm=TRUE))
lab_withvars <- subna(lab_withvars, "lots_of_patents", 0)
lab_withvars <- subna(lab_withvars, "n_employees", mean(lab_withvars$n_employees, na.rm=TRUE))
lab_withvars <- subna(lab_withvars, "intangible_fa", mean(lab_withvars$intangible_fa, na.rm=TRUE))
lab_withvars <- subna(lab_withvars, "patent_ct", 1)
lab_withvars <- subna(lab_withvars, "n_subsidiaries", 0)
lab_withvars <- subna(lab_withvars, "has_first_name", 0)
lab_withvars <- subna(lab_withvars, "is_not_person_hat", 1)
lab_withvars <- subna(lab_withvars, "is_person_hat", 0)
lab_withvars <- subna(lab_withvars, "certain_not_person", 0)
lab_withvars <- subna(lab_withvars, "is_person", 0)
lab_withvars <- subna(lab_withvars, "maybe_foreign_legal", 0)
lab_withvars <- subna(lab_withvars, "has_legal_in", 0)

countna(lab_withvars)

lab_withvars$jw2_name <- inter(lab_withvars$jw_name_dist, lab_withvars$jw_name_dist)
lab_withvars$lev2_name <- inter(lab_withvars$lev_name_dist, lab_withvars$lev_name_dist)
lab_withvars$jw2_legal <- inter(lab_withvars$legal_jw, lab_withvars$legal_jw)
lab_withvars$jw2_uncommon <- inter(lab_withvars$name_less_common_jw, lab_withvars$name_less_common_jw)
lab_withvars$geo_string <- inter(lab_withvars$jw_name_dist, lab_withvars$geo_dist)
lab_withvars$int_patents <- inter(lab_withvars$patent_ct, lab_withvars$intangible_fa)

interactions <- c("jw2_name", "lev2_name", "jw2_legal", "jw2_uncommon", "geo_string")

#lab_withvars$geo_string[is.na(lab_withvars$geo_string)] <- mean(lab_withvars$geo_string, na.rm=TRUE)
#lab_withvars$int_patents[is.na(lab_withvars$int_patents)] <- 0
#lab_withvars$name_abbreviated[is.na(lab_withvars$name_abbreviated)] <- 1
#lab_withvars$has_legal_in[is.na(lab_withvars$has_legal_in)] <- 0
#lab_withvars$maybe_foreign_legal[is.na(lab_withvars$maybe_foreign_legal)] <- 0
#lab_withvars$is_not_person_hat[is.na(lab_withvars$is_not_person_hat)] <- 1
#lab_withvars$has_first_name[is.na(lab_withvars$has_first_name)] <- 0
#lab_withvars$lots_of_patents[is.na(lab_withvars$lots_of_patents)] <- 0

# training data preparation for the models
# list of non-variables
entnames <- c("name_less_common_ps", "name_less_common_am", "metaphone_am", "metaphone_ps")

# important! appears in default_processing function
exclude_these <- c(ids, entnames, true_match, "is_not_person_hat", "ipc_code", "year")

# divide data into training and validation
patstat_ids <- as.character(unique(unlist(lab_withvars$patstat_id, use.names = FALSE)))

random_sample <- sample(patstat_ids, length(patstat_ids)/2)
valid_sample <- lab_withvars[!(lab_withvars[,"patstat_id"] %in% random_sample), "patstat_id"]

# no mods training data
sample_rows <- which(lab_withvars[,"patstat_id"] %in% as.character(random_sample))
valid_rows <- which(lab_withvars[,"patstat_id"] %in% as.character(valid_sample))
#design_matrix <- lab_withvars[lab_withvars[,"patstat_id"] %in% random_sample,]
design_matrix <- lab_withvars
design_matrix <- design_matrix[,!names(design_matrix) %in% exclude_these]
    
model_matrix_first <- model.matrix(~ . + (jw_name_dist + lev_name_dist + legal_jw +
            name_less_common_jw + metaphone_jw + person_class +
            avg_freq_am + qavg_freq_am + 
            geo_dist + oprevenue + sector_sim_max + sector_sim_sum + n_subsidiaries +
            n_employees + intangible_fa + min_jw_of_alt + max_sec_of_alt +
            is_matchable + patent_ct)^2 
                                     + interaction(country, naics_2007)
                                     + interaction(country, bracket)
                                     + interaction(country, geo_cat)
                                     #+ interaction(country, is_matchable)
                                               , data=design_matrix)

# country  dummies: used to create interactions with is_matchable
int_matchable_country <- model.matrix(~. , data = data.frame(design_matrix$country))
    
for(nn in 1:dim(int_matchable_country)[2]){
        int_matchable_country[,nn] <- int_matchable_country[,nn] * design_matrix$is_matchable
}
int_matchable_country <- int_matchable_country[,-1]
colnames(int_matchable_country) <- sapply(colnames(int_matchable_country), 
                                        function(x) paste0("matchable-country-",substr(x, nchar(x)-1, nchar(x))))
    
model_matrix_final <- cbind(model_matrix_first, int_matchable_country)
    
y <- lab_withvars[sample_rows, "true_match"]

# LASSO MODEL (takes care of dummies and of constants automatically)
model <- glmnet(model_matrix_final[sample_rows,], y, 
                family="binomial") 

model_matrix_v <- model_matrix_final[valid_rows,]
yv <- lab_withvars[valid_rows, "true_match"]

#patstat_ids <- unique(unlist(xv$patstat_id, use.names = FALSE))

# LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO 
# LOOP on coefficient sequence
        # survived vars matrix and coeff vector, incl intercept
fscore <- rep(NA, dim(model$beta)[2])
cutoffs <- rep(NA, dim(model$beta)[2])
    
for(i in 1:dim(model$beta)[2]){
        print(paste(i, "of", dim(model$beta)[2]))
        x_sel <- cbind(1, model_matrix_v[,names(which(model$beta[,i] != 0))])
        b_sel <- c(model$a0[i], model$beta[,i][model$beta[,i] != 0])
    
        xb_sel <- as.matrix(x_sel) %*% as.matrix(b_sel)
        lasso_p_hat <- 1/(1+exp(-xb_sel))
        #print(lasso_p_hat)
        maxprobs_only <- lasso_p_hat
        xv_aug <- cbind(lab_withvars[valid_rows,ids], lasso_p_hat, maxprobs_only)

        for(j in valid_sample){
            subset_max <- max(xv_aug[xv_aug$patstat_id == j,"lasso_p_hat"])
            xv_aug[xv_aug$patstat_id == j & xv_aug$lasso_p_hat<subset_max,"maxprobs_only"] <- 0
        }
        cpred1 <- prediction(xv_aug$maxprobs_only, yv)
        auc_p <- as.numeric(performance(cpred1,"f")@y.values[[1]])
        f_cutoff <- as.numeric(performance(cpred1,"f")@x.values[[1]])
        auc_p[is.na(auc_p)] <- 0
        f_idx <- which(auc_p == max(auc_p))
        fscore[i] <- max(auc_p)
        print(fscore[i])
        cutoffs[i] <- f_cutoff[f_idx[1]]
}

best_n <- which(fscore == max(fscore, na.rm=TRUE))[1]
print(best_n)
best_fscore <- fscore[best_n][1]
best_cutoff <- cutoffs[best_n][1]
names(best_cutoff) <- "best_cutoff"

besta0 <- model$a0[best_n]
bestlambda <- model$lambda[best_n]

survived_vars <- model$beta[,best_n][model$beta[,best_n]!=0]
survived_vars_names <- names(model$beta[,best_n][model$beta[,best_n]!=0])
write.table(t(c(best_cutoff, survived_vars)), 
            "/home/desktop/patstat_data/all_code/remerge/R/guesses/survived_variables_wrds_v4.csv", sep='\t', row.names=FALSE)
    
#survived_xv <- as.matrix(model_matrix_v[,survived_vars_names]) # these two lines same as xb
#estim_xb <- cbind(1, survived_xv) %*% c(besta0, survived_vars)

# colnames(test)[grepl("country", colnames(test))][1:100]

flist <- Sys.glob('/home/desktop/patstat_data/all_code/remerge/regression_data/whole/*v4.tsv')
for(fname in flist[7:length(flist)]){
    #fname <- Sys.glob('/home/desktop/patstat_data/all_code/remerge/regression_data/whole/*v4.tsv')[1]
    cat(fname , '\n')
    cat('loading...', '\n')
    datafile <- read.table(fname, sep='\t', header=TRUE)
    cat('fitting...', '\n')
    listfiles <- split(datafile, 1:ceiling(dim(datafile)[1]/50000))
    phats <- c()
    for(inlist in listfiles){
        cat('processing new chunk with dimension ', dim(inlist), '\n')
        new_x <- default_processing(inlist)
        new_predicts <- predict(model, new_x, s = bestlambda)
        new_estims <- 1/(1+exp(-new_predicts))
        phats <- c(phats, new_estims)
        gc()
    }
    results <- cbind(datafile[,c("patstat_id", "company_id", "country")], phats)
    colnames(results) <- c("patstat_id", "company_id", "country", "phat")
    #results <- results[order(results[,"patstat_id"], -results[,"phat"]),]
    cat('aggregating and saving' , '\n')
    groupresults <- data.table(results[order(results[,"patstat_id"], -results[,"phat"]),])
    aggresults <- groupresults[, list(company_id = company_id[1], 
                                country = country[1], 
                                phat = phat[1]), 
                          by = c("patstat_id")]
    ctry <- aggresults$country[1]
    write.table(aggresults, 
                paste0('/home/desktop/patstat_data/all_code/remerge/matched_by_remerge/',
                            'match_by_R_', ctry, '.csv'), 
                sep='\t', 
                row.names=FALSE )
}

estim_xb <- predict(model, model_matrix_v, s = bestlambda) # this finds xb only
estim_probs <- 1/(1+exp(-estim_xb))

maxprobs_only <- estim_probs
xv_aug <- cbind(lab_withvars[valid_rows,ids], estim_probs, maxprobs_only)

print(best_cutoff)
for(i in valid_sample){
        subset_max <- max(xv_aug[xv_aug$patstat_id == i,"estim_probs"])
        xv_aug[xv_aug$patstat_id == i & xv_aug$estim_probs<subset_max,"maxprobs_only"] <- 0
}

model_prediction <- performance(prediction(xv_aug[,"maxprobs_only"], yv), "prec", "rec")
prec_rec_table <- cbind(as.numeric(model_prediction@y.values[[1]]),
                            as.numeric(model_prediction@x.values[[1]]),
                            as.numeric(model_prediction@alpha.values[[1]]))
colnames(prec_rec_table) <- c("precision", "recall", "cutoff")

y_hat <- rep(0, dim(xv_aug)[1])
y_hat[xv_aug[,"maxprobs_only"] >= best_cutoff] <- 1
    
output_matrix <- cbind(lab_withvars[valid_sample,ids], 
                            xv_aug$estim_probs,
                            xv_aug$maxprobs_only, 
                            lab_withvars[valid_sample,"country"],
                            y_hat,
                            yv)
                                       
colnames(output_matrix) <- c(ids, "score", "maximal_score", "country", "y_estimated", "y_true")
#write.table(output_matrix, paste0("guesses/matching_regression_out/","fR_estim_probs_",country,".csv"), sep='\t', row.names=FALSE)
write.table(output_matrix, "/home/desktop/patstat_data/all_code/remerge/R/guesses/matching_regression_out/matched_validation_wrds_v3.csv", sep='\t', row.names=FALSE)
    
