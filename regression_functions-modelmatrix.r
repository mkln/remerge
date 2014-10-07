inter <- function(list1, list2){
    return(list1 * list2)
}


#interact_ipc_country <- function(ipc_matrix, country_factor){
#    large_matrix <- c()
#    for(col_n in 1:dim(ipc_matrix)[2]){
#        small_matrix <- model.matrix(~. , data = data.frame(ipc_matrix[,col_n], country_factor))
#        large_matrix <- cbind(large_matrix, small_matrix)
#    }
#    return(large_matrix)
#}


classify_performance <- function(twocolsmatrix){
    ytrue <- twocolsmatrix[,1]
    yhat <- twocolsmatrix[,2]
    tp <- length(which(ytrue[which(yhat == 1)] == 1))
    fp <- length(which(ytrue[which(yhat == 1)] == 0))
    fn <- length(which(ytrue[which(yhat == 0)] == 1))

    recall <- tp / (tp + fn)
    precision <- tp / (tp + fp)
        
    fscore <- 2*(precision * recall)/(precision + recall)
    if(is.na(fscore)){ fscore <- 0 }

    results <- c(tp, fp, fn, precision, recall, fscore)
    names(results) <- c("true_positives", "false_positives", "false_negatives",
                        "precision", "recall", "fscore")
    return(results)
    
}


max_f_score <- function(trues, scores){
    model_prediction <- performance(prediction(scores, trues), "f")
    fscores <- model_prediction@y.values[[1]]
    fscores[is.na(fscores)] <- 0

    results <- list()
    results[[1]] <- max(fscores)

    cutoffs <- as.numeric(model_prediction@x.values[[1]])
    best_cutoff <- cutoffs[which(fscores == max(fscores))[1]]

    results[[2]] <- best_cutoff

    model_prediction <- performance(prediction(scores, trues), "prec", "rec")
    prec_rec_table <- c(as.numeric(model_prediction@y.values[[1]])[which(fscores == max(fscores))[1]],
                            as.numeric(model_prediction@x.values[[1]])[which(fscores == max(fscores))[1]],
                            as.numeric(model_prediction@alpha.values[[1]])[which(fscores == max(fscores))[1]])
    names(prec_rec_table) <- c("precision", "recall", "cutoff")

    results[[3]] <- prec_rec_table

names(results) <- c("best_fscore", "best_cutoff", "precision and recall of best")
return(results)

}

optimizer_of_f <- function(trues, maxscore, ismatchable){
    expons <- seq(0,20,0.1)
    fscores <- rep(0, length(expons))
    for(exx in 1:length(expons)){
        fscores[exx] <- max_f_score(trues, maxscore*(ismatchable^expons[exx]))[[1]]
    }  
    return(c(max(fscores), expons[which(fscores == max(fscores))[1]]))
    
}

multi_dummy <- function(this_df, this_id, this_col, substr_n){
    reduced_df <- unique(this_df[, c(this_id, this_col)])
    unique_val <- unique( substr(
                        unique(unlist(strsplit(gsub("\\*","=", reduced_df[,this_col]), split="==")))
                        , 1,substr_n)
                        )
    unique_val <- unique_val[unique_val != ""]
    dummy_vec <- array(0, dim=c( length(unique_val) ))
    names(dummy_vec) <- unique_val

    dummy_mat <- as.data.frame(t(sapply(reduced_df[,this_col], function(concat_val) {
                                    row_val <- unique(substr(unlist(strsplit(gsub("\\*","=",concat_val), split="==")), 1,substr_n))
                                    matching <- unlist(intersect(row_val, unique_val))
                                    matching <- matching[matching!=""]
                                    dummy_vec[matching] <- 1
                                    return(dummy_vec)
                                        })))
    
    if(this_col == "year"){
        colnames(dummy_mat) <- sapply(colnames(dummy_mat), function(x) {return(paste0("X", x))})
    }
    dummy_mat[,this_id] <- reduced_df[,this_id]
    return_df <- left_join(this_df, dummy_mat)
    return(return_df)
}

subna <- function(d, column, value){
    rd <- d
    rd[is.na(d[,column]), column] <- value
    return( rd ) 
} 

countna <- function(d){
    with_na <- c()
    for(i in 1:dim(d)[2]){
        if(length(which(is.na(d[,i]))) != 0){
                with_na <- c(with_na, colnames(d)[i])
                cat(c(colnames(d)[i], " ", length(which(is.na(d[,i]))), "\n"))
        }
    }
    if(length(with_na) > 0){
        cat("Warning: some columns still have missing values! \n")
        } 
    return(with_na)
}

default_processing <- function(dataframe){
    retdf <- dataframe
    
    fakerow <- retdf[1,]
    fakerow[,"country"] <- "AA"
    retdf <- rbind(fakerow, retdf)
    
    retdf <- retdf[!is.na(retdf$jw_name_dist),]
    retdf <- multi_dummy(retdf, "patstat_id", "ipc_code", 3)
    retdf <- multi_dummy(retdf, "patstat_id", "year", 4)
    
    gc()
    
    retdf$naics_2007[is.na(retdf$naics_2007)] <- "missing"
    retdf$oprevenue[is.na(retdf$oprevenue)] <- 0
    retdf[,"naics_2007"] <- as.factor(retdf[,"naics_2007"])
    retdf[,"bracket"] <- as.factor(retdf[,"bracket"])
    retdf[,"geo_cat"] <- as.factor(retdf[,"geo_cat"])
    retdf <- subna(retdf, "geo_dist", mean(retdf$geo_dist, na.rm=TRUE))
    retdf <- subna(retdf, "lots_of_patents", 0)
    retdf <- subna(retdf, "n_employees", mean(retdf$n_employees, na.rm=TRUE))
    retdf <- subna(retdf, "intangible_fa", mean(retdf$intangible_fa, na.rm=TRUE))
    retdf <- subna(retdf, "patent_ct", 1)
    retdf <- subna(retdf, "n_subsidiaries", 0)
    retdf <- subna(retdf, "has_first_name", 0)
    retdf <- subna(retdf, "is_not_person_hat", 1)
    retdf <- subna(retdf, "is_person_hat", 0)
    retdf <- subna(retdf, "certain_not_person", 0)
    retdf <- subna(retdf, "is_person", 0)
    retdf <- subna(retdf, "maybe_foreign_legal", 0)
    retdf <- subna(retdf, "has_legal_in", 0)

    retdf$jw2_name <- inter(retdf$jw_name_dist, retdf$jw_name_dist)
    retdf$lev2_name <- inter(retdf$lev_name_dist, retdf$lev_name_dist)
    retdf$jw2_legal <- inter(retdf$legal_jw, retdf$legal_jw)
    retdf$jw2_uncommon <- inter(retdf$name_less_common_jw, retdf$name_less_common_jw)
    retdf$geo_string <- inter(retdf$jw_name_dist, retdf$geo_dist)
    retdf$int_patents <- inter(retdf$patent_ct, retdf$intangible_fa)
    
    retdf <- retdf[,!names(retdf) %in% exclude_these]
    
    which_factors <- colnames(retdf)[sapply(retdf, is.factor)]
    for(f in which_factors){
        cat(paste(f, nlevels(retdf[,f]), '\n'))
    }
    
    mm1 <- model.matrix.default(~ . + (jw_name_dist + lev_name_dist + legal_jw +
            name_less_common_jw + metaphone_jw + person_class +
            avg_freq_am + qavg_freq_am + 
            geo_dist + oprevenue + sector_sim_max + sector_sim_sum + n_subsidiaries +
            n_employees + intangible_fa + min_jw_of_alt + max_sec_of_alt +
            is_matchable + patent_ct + jw2_name + lev2_name)^2 +
                                        interaction(country, naics_2007)
                                     + interaction(country, bracket)
                                     + interaction(country, geo_cat)
                                     #+ interaction(country, is_matchable)
                                               , data=retdf)
                                               
    
    # country  dummies: used to create interactions with is_matchable
    int_matchable_country <- model.matrix(~. , data = data.frame(retdf$country))
        
    for(nn in 1:dim(int_matchable_country)[2]){
            int_matchable_country[,nn] <- int_matchable_country[,nn] * retdf$is_matchable
    }
    int_matchable_country <- int_matchable_country[,-1]
    names(int_matchable_country) <- sapply(colnames(int_matchable_country), 
                                            function(x) paste0("matchable-country-",substr(x, nchar(x)-1, nchar(x))))
        
    retdf <- cbind(mm1, int_matchable_country)
    
    # create a 0-filled dataframe with same columns as model matrix
    finaldf <- matrix(0, nrow=dim(retdf)[1], ncol=dim(model_matrix_final)[2])
    colnames(finaldf) <- colnames(model_matrix_final)
    finaldf <- as.data.frame(finaldf)
    
    # find which columns in this data are also in the fitted model
    available_cols <- intersect(colnames(model_matrix_final), colnames(retdf))
    
    # update values for those columns
    finaldf[,available_cols] <- retdf[,available_cols]
    
return(as.matrix(finaldf[2:dim(finaldf)[1],colnames(model_matrix_final)]))

}
