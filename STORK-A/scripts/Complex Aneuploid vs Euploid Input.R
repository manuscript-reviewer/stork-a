library(openxlsx)
library(dataPreparation)
library(caret)
library(epiR)
library(ROSE)
library(ROCR)
library(zoo)
library(dplyr)

######
Z1 <- 'img_110'
Z2 <- 'img_110_age'
Z3 <- 'img_110_morpho'
Z4 <- 'img_110_BS'
Z5 <- 'img_110_QUAL'
Z6 <- 'img_110_age_QUAL'
Z7 <- 'img_110_age_morpho'
Z8 <- 'img_110_age_BS'
Z9 <- 'img_110_age_morpho_BS'
Z10 <- 'img_110_age_morpho_QUAL'
############################################

z_set <- c(Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10)

for (A in 1:length(z_set)) {
  feature_selection <- z_set[A]
  
  #######################################
  new_folder <- paste0(feature_selection, '_CA-EUP')
  quick <- '~/Desktop/Github/Data/Complex Aneupoid vs Euploid/'
  
  folder_path <- paste0(quick, new_folder)
  
  dir.create(folder_path)
  train_write <- paste0(folder_path, "/training.txt")
  val_write <- paste0(folder_path, "/validation.txt")
  test_write <- paste0(folder_path, "/test.txt")
  
  #DATLOADER ###################################
  #load complete data
  data <-
    read.xlsx("~/Desktop/Github/Data/PGTA-Data.xlsx")#spreadsheet with sample names, PGT results, morphokinetics, morphological assesment
  data <-
    data[,-c(2:4, 10, 23:25, 27)] #keep only columns of interest
  
  imgs <-
    read.xlsx('~/Desktop/Github/Data/image_IDs.xlsx', colNames = T) #spreadsheet with sample image file names at 110-hours
  
  data$img_file <- imgs$File[match(data$SUBJECT_NO, imgs$ID)]
  
  data$age_group <-
    ifelse(data$EGG_AGE < 35 ,
           "age1",
           ifelse(
             data$EGG_AGE < 38,
             "age2",
             ifelse(data$EGG_AGE >= 41, "age4", "age3")
           ))
  colnames(data)[15] <- 't9'
  fulldata <- data
  
  cols <-
    c('ab_or_norm',
      'PGD_RESULT',
      'BX_DAY',
      'BS',
      'Expansion',
      'ICM',
      'TE',
      'age_group')
  data[cols] <- lapply(data[cols], as.factor)
  
  #Make working dataset
  data_work <- data
  
  #hot encode features with factors
  encoding_BS <-
    build_encoding(data_work, cols = c('BS'), verbose = F)
  data_work <-
    one_hot_encoder(data_work,
                    encoding = encoding_BS,
                    drop = T,
                    verbose = F)
  
  #length of dataset with blastocyst score
  bs_length <- 25:ncol(data_work) - 8
  #length of dataset with blastocyst grade
  qual1_length <- ncol(data_work) + 1
  
  encoding_ICM <-
    build_encoding(data_work, cols = c('ICM'), verbose = F)
  data_work <-
    one_hot_encoder(data_work,
                    encoding = encoding_ICM,
                    drop = T,
                    verbose = F)
  
  encoding_TE <-
    build_encoding(data_work, cols = c('TE'), verbose = F)
  data_work <-
    one_hot_encoder(data_work,
                    encoding = encoding_TE,
                    drop = T,
                    verbose = F)
  
  encoding_Expan <-
    build_encoding(data_work,
                   cols = c('Expansion'),
                   verbose = F)
  data_work <-
    one_hot_encoder(data_work,
                    encoding = encoding_Expan,
                    drop = T,
                    verbose = F)
  
  qual1_length <- qual1_length - 3
  qual_length <- qual1_length:ncol(data_work) - 5
  
  data_work <- data.frame(data_work)
  
  to_end <- as.numeric(ncol(data_work))
  imp_data <- data_work[, c(1, 21, 4, 3, 2, 7:17, 22:to_end)]
  imp_data <- as.data.frame(imp_data)
  imp_data <- data.frame(imp_data)
  
  
  # IMPUTE TRAINING AND VALIDATION SETS ###########################
  set.seed(8699)
  
  imp_data$BS <-
    fulldata$BS[match(imp_data$SUBJECT_NO, fulldata$SUBJECT_NO)]
  imp_data$PGD_RESULT <-
    fulldata$PGD_RESULT[match(imp_data$SUBJECT_NO, fulldata$SUBJECT_NO)]
  
  EUP_data <- imp_data[imp_data$PGD_RESULT == 'EUP', ]
  EUP_index <-
    createDataPartition(EUP_data$ab_or_norm,
                        p = 0.70,
                        list = FALSE,
                        times = 1)
  EUP_train <- EUP_data[EUP_index, ]
  EUP_val <- EUP_data[-EUP_index, ]
  
  CxA_data <- imp_data[imp_data$PGD_RESULT == 'CxA', ]
  CxA_index <-
    createDataPartition(CxA_data$ab_or_norm,
                        p = 0.70,
                        list = FALSE,
                        times = 1)
  CxA_train <- CxA_data[CxA_index, ]
  CxA_val <- CxA_data[-CxA_index, ]
  
  training <- rbind(CxA_train, EUP_train)
  class_num_train <- table(training$ab_or_norm)
  decide <- class_num_train[1] < class_num_train[2]
  bal_num <- as.numeric(max(class_num_train) - min(class_num_train))
  
  if (decide == T) {
    rand_dbl <-
      CxA_train[sample(nrow(CxA_train), bal_num, replace = F), ]
    training <- rbind(CxA_train, EUP_train, rand_dbl)
  } else if (decide == F) {
    rand_dbl <-
      EUP_train[sample(nrow(EUP_train), bal_num, replace = F), ]
    training <- rbind(ANU_train, EUP_train, rand_dbl)
  }
  
  training <- training[, -c(ncol(training))]
  
  validation <- rbind(CxA_val, EUP_val)
  validation <- validation[, -c(ncol(validation))]
  
  
  missing_data <- function(x) {
    colSums(is.na(x))
  }
  
  impute_median <- function(dataset, mid_data) {
    x = data.frame(which(missing_data(dataset) > 0))[, 1]
    for (i in x) {
      idx = which(is.na(dataset[, i]))
      dataset[idx, ][i] <- median(mid_data[, i], na.rm = T)
    }
    return(dataset)
  }
  
  
  ################IMPUTE BS for training
  
  ANU3 <-
    training[training$BS == 3 & training$ab_or_norm == "Abnormal", ]
  ANU4 <-
    training[training$BS == 4 & training$ab_or_norm == "Abnormal", ]
  ANU5 <-
    training[training$BS == 5 & training$ab_or_norm == "Abnormal", ]
  ANU6 <-
    training[training$BS == 6 & training$ab_or_norm == "Abnormal", ]
  ANU7 <-
    training[training$BS == 7 & training$ab_or_norm == "Abnormal", ]
  ANU8 <-
    training[training$BS == 8 & training$ab_or_norm == "Abnormal", ]
  ANU9 <-
    training[training$BS == 9 & training$ab_or_norm == "Abnormal", ]
  ANU10 <-
    training[training$BS == 10 & training$ab_or_norm == "Abnormal", ]
  ANU11 <-
    training[training$BS == 11 & training$ab_or_norm == "Abnormal", ]
  ANU12 <-
    training[training$BS == 12 & training$ab_or_norm == "Abnormal", ]
  ANU13 <-
    training[training$BS == 13 & training$ab_or_norm == "Abnormal", ]
  ANU14 <-
    training[training$BS == 14 & training$ab_or_norm == "Abnormal", ]
  ANU15 <-
    training[training$BS == 15 & training$ab_or_norm == "Abnormal", ]
  ANU16 <-
    training[training$BS == 16 & training$ab_or_norm == "Abnormal", ]
  ANU17 <-
    training[training$BS == 17 & training$ab_or_norm == "Abnormal", ]
  
  EUP3 <-
    training[training$BS == 3 & training$ab_or_norm == "Normal", ]
  EUP4 <-
    training[training$BS == 4 & training$ab_or_norm == "Normal", ]
  EUP5 <-
    training[training$BS == 5 & training$ab_or_norm == "Normal", ]
  EUP6 <-
    training[training$BS == 6 & training$ab_or_norm == "Normal", ]
  EUP7 <-
    training[training$BS == 7 & training$ab_or_norm == "Normal", ]
  EUP8 <-
    training[training$BS == 8 & training$ab_or_norm == "Normal", ]
  EUP9 <-
    training[training$BS == 9 & training$ab_or_norm == "Normal", ]
  EUP10 <-
    training[training$BS == 10 & training$ab_or_norm == "Normal", ]
  EUP11 <-
    training[training$BS == 11 & training$ab_or_norm == "Normal", ]
  EUP12 <-
    training[training$BS == 12 & training$ab_or_norm == "Normal", ]
  EUP13 <-
    training[training$BS == 13 & training$ab_or_norm == "Normal", ]
  EUP14 <-
    training[training$BS == 14 & training$ab_or_norm == "Normal", ]
  EUP15 <-
    training[training$BS == 15 & training$ab_or_norm == "Normal", ]
  EUP16 <-
    training[training$BS == 16 & training$ab_or_norm == "Normal", ]
  EUP17 <-
    training[training$BS == 17 & training$ab_or_norm == "Normal", ]
  
  ANU3 <- impute_median(ANU3, ANU3)
  ANU4 <- impute_median(ANU4, ANU4)
  ANU5 <- impute_median(ANU5, ANU5)
  ANU6 <- impute_median(ANU6, ANU6)
  ANU7 <- impute_median(ANU7, ANU7)
  ANU8 <- impute_median(ANU8, ANU8)
  ANU9 <- impute_median(ANU9, ANU9)
  ANU10 <- impute_median(ANU10, ANU10)
  ANU11 <- impute_median(ANU11, ANU11)
  ANU12 <- impute_median(ANU12, ANU12)
  ANU13 <- impute_median(ANU13, ANU13)
  ANU14 <- impute_median(ANU14, ANU14)
  ANU15 <- impute_median(ANU15, ANU15)
  ANU16 <- impute_median(ANU16, ANU16)
  ANU17 <- impute_median(ANU17, ANU17)
  
  EUP3 <- impute_median(EUP3, EUP3)
  EUP4 <- impute_median(EUP4, EUP4)
  EUP5 <- impute_median(EUP5, EUP5)
  EUP6 <- impute_median(EUP6, EUP6)
  EUP7 <- impute_median(EUP7, EUP7)
  EUP8 <- impute_median(EUP8, EUP8)
  EUP9 <- impute_median(EUP9, EUP9)
  EUP10 <- impute_median(EUP10, EUP10)
  EUP11 <- impute_median(EUP11, EUP11)
  EUP12 <- impute_median(EUP12, EUP12)
  EUP13 <- impute_median(EUP13, EUP13)
  EUP14 <- impute_median(EUP14, EUP14)
  EUP15 <- impute_median(EUP15, EUP15)
  EUP16 <- impute_median(EUP16, EUP16)
  EUP17 <- impute_median(EUP17, EUP17)
  
  
  #recombine dataset
  training_impute <-
    rbind(
      ANU3,
      ANU4,
      ANU5,
      ANU6,
      ANU7,
      ANU8,
      ANU9,
      ANU10,
      ANU11,
      ANU12,
      ANU13,
      ANU14,
      ANU15,
      ANU16,
      ANU17,
      EUP3,
      EUP4,
      EUP5,
      EUP6,
      EUP7,
      EUP8,
      EUP9,
      EUP10,
      EUP11,
      EUP12,
      EUP13,
      EUP14,
      EUP15,
      EUP16,
      EUP17
    )
  
  training_impute <- training_impute[, -c(ncol(training))]
  
  #VALIDATION
  #use same medians as in training making sure that new medians arent replacing old medians
  ###becasue we wont know if a sample is abnormal or normal, we take medians based off of all bs
  all_BS3 <- training[training$BS == 3  , ]
  all_BS4 <- training[training$BS == 4  , ]
  all_BS5 <- training[training$BS == 5  , ]
  all_BS6 <- training[training$BS == 6  , ]
  all_BS7 <- training[training$BS == 7  , ]
  all_BS8 <- training[training$BS == 8  , ]
  all_BS9 <- training[training$BS == 9  , ]
  all_BS10 <- training[training$BS == 10  , ]
  all_BS11 <- training[training$BS == 11  , ]
  all_BS12 <- training[training$BS == 12  , ]
  all_BS13 <- training[training$BS == 13  , ]
  all_BS14 <- training[training$BS == 14  , ]
  all_BS15 <- training[training$BS == 15  , ]
  all_BS16 <- training[training$BS == 16  , ]
  all_BS17 <- training[training$BS == 17  , ]
  
  ###
  BS3_val <- validation[validation$BS == 3, ]
  BS4_val <- validation[validation$BS == 4, ]
  BS5_val <- validation[validation$BS == 5, ]
  BS6_val <- validation[validation$BS == 6, ]
  BS7_val <- validation[validation$BS == 7, ]
  BS8_val <- validation[validation$BS == 8, ]
  BS9_val <- validation[validation$BS == 9, ]
  BS10_val <- validation[validation$BS == 10, ]
  BS11_val <- validation[validation$BS == 11, ]
  BS12_val <- validation[validation$BS == 12, ]
  BS13_val <- validation[validation$BS == 13, ]
  BS14_val <- validation[validation$BS == 14, ]
  BS15_val <- validation[validation$BS == 15, ]
  BS16_val <- validation[validation$BS == 16, ]
  BS17_val <- validation[validation$BS == 17, ]
  
  BS3_val <- impute_median(BS3_val, all_BS3)
  BS4_val <- impute_median(BS4_val, all_BS4)
  BS5_val <- impute_median(BS5_val, all_BS5)
  BS6_val <- impute_median(BS6_val, all_BS6)
  BS7_val <- impute_median(BS7_val, all_BS7)
  BS8_val <- impute_median(BS8_val, all_BS8)
  BS9_val <- impute_median(BS9_val, all_BS9)
  BS10_val <- impute_median(BS10_val, all_BS10)
  BS11_val <- impute_median(BS11_val, all_BS11)
  BS12_val <- impute_median(BS12_val, all_BS12)
  BS13_val <- impute_median(BS13_val, all_BS13)
  BS14_val <- impute_median(BS14_val, all_BS14)
  BS15_val <- impute_median(BS15_val, all_BS15)
  BS16_val <- impute_median(BS16_val, all_BS16)
  BS17_val <- impute_median(BS17_val, all_BS17)
  
  #make training set
  validation_impute <-
    rbind(
      BS3_val,
      BS4_val,
      BS5_val,
      BS6_val,
      BS7_val,
      BS8_val,
      BS9_val,
      BS10_val,
      BS11_val,
      BS12_val,
      BS13_val,
      BS14_val,
      BS15_val,
      BS16_val,
      BS17_val
    )
  
  validation_impute <- validation_impute[, -c(ncol(training))]
  
  
  
  ######################FEATURE SELECTION#########################
  #select which features to use
  if (feature_selection == "img_110") {
    feature_set <- c(1, 3)
  } else if (feature_selection == "img_110_age") {
    feature_set <- c(1, 3, 5)
  } else if (feature_selection == "img_110_morpho") {
    feature_set <- c(1, 3, 6:16)
  } else if (feature_selection == "img_110_BS") {
    feature_set <- c(1, 3, bs_length)
  } else if (feature_selection == "img_110_QUAL") {
    feature_set <- c(1, 3, qual_length)
  } else if (feature_selection == "img_110_age_morpho") {
    feature_set <- c(1, 3, 5:16)
  } else if (feature_selection == "img_110_age_BS") {
    feature_set <- c(1, 3, 5, bs_length)
  } else if (feature_selection == "img_110_age_QUAL") {
    feature_set <- c(1, 3, 5, qual_length)
  } else if (feature_selection == "img_110_age_morpho_BS") {
    feature_set <- c(1, 3, 5:16, bs_length)
  } else if (feature_selection == "img_110_age_morpho_QUAL") {
    feature_set <- c(1, 3, 5:16, qual_length)
  } else if (feature_selection == "img_110_age_pred_BS") {
    feature_set <- c(1, 3)
  }
  
  #################
  
  # FINAL CREATION OF TRAIN AND VALIDATION #########
  training_impute <- training_impute[, feature_set]
  validation_impute <- validation_impute[, feature_set]
  
  input_names <- c(colnames(training_impute[-c(1, 2)]))
  
  #make sure to scale approriately
  scales <-
    build_scales(dataSet = training_impute,
                 cols = input_names,
                 verbose = F)
  training_impute <-
    fastScale(training_impute, scales = scales, verbose = F)
  validation_impute <-
    fastScale(validation_impute, scales = scales, verbose = F)
  
  training_impute <- data.frame(training_impute)
  validation_test_impute <- data.frame(validation_impute)
  validation_test_impute$PGD <-
    fulldata$PGD_RESULT[match(validation_test_impute$SUBJECT_NO, fulldata$SUBJECT_NO)]
  
  EUP_VT <-
    validation_test_impute[validation_test_impute$PGD == "EUP", ]
  EUP_VT_index <-
    createDataPartition(EUP_VT$ab_or_norm,
                        p = 0.50,
                        list = FALSE,
                        times = 1)
  EUP_VT_val <- EUP_VT[EUP_VT_index, ]
  EUP_VT_test <- EUP_VT[-EUP_VT_index, ]
  
  CxA_VT <-
    validation_test_impute[validation_test_impute$PGD == "CxA", ]
  CxA_VT_index <-
    createDataPartition(CxA_VT$ab_or_norm,
                        p = 0.50,
                        list = FALSE,
                        times = 1)
  CxA_VT_val <- CxA_VT[CxA_VT_index, ]
  CxA_VT_test <- CxA_VT[-CxA_VT_index, ]
  
  validation_impute <- rbind(EUP_VT_val, CxA_VT_val)
  validation_impute <- validation_impute[, -c(ncol(validation_impute))]
  test_impute <- rbind(EUP_VT_test, CxA_VT_test)
  test_impute <- test_impute[, -c(ncol(test_impute))]
  
  make_input <- function(x) {
    x$Label <- ifelse(x$ab_or_norm == 'Abnormal', 0, 1)
    x$img_file <- data$img_file[match(x$SUBJECT_NO, data$SUBJECT_NO)]
    fx <- c("img_file", 'Label')
    fy <- input_names
    x <- x[, c(fx, fy)]
    x <- x[colSums(!is.na(x)) > 0]
    
    colnames(x)[1:2] <- c("img", "Label")
    
    if (feature_selection != 'img_110') {
      tot_clin_data = ncol(x) - 2
      colnames(x)[3:ncol(x)] <- c(1:tot_clin_data)
    }
    
    x$order <- 0:(nrow(x) - 1)
    num_feat = (ncol(x) - 1)
    num_col = ncol(x)
    x <- x[, c(num_col, 1:num_feat)]
    colnames(x)[1] <- ""
    return(x)
  }
  
  
  train_data <- make_input(training_impute)
  train_data <- train_data[sample(nrow(train_data)), ]
  train_data[, 1] <- 0:(nrow(train_data) - 1)
  
  val_data <- make_input(validation_impute)
  val_data <- val_data[sample(nrow(val_data)), ]
  val_data[, 1] <- 0:(nrow(val_data) - 1)
  
  test_data <- make_input(test_impute)
  test_data <- test_data[sample(nrow(test_data)), ]
  test_data[, 1] <- 0:(nrow(test_data) - 1)
  
  ###########################################
  
  #outputs

    write.table(
      train_data,
      train_write,
      sep = "\t",
      row.names = F,
      col.names = T,
      quote = FALSE
    )
    write.table(
      val_data,
      val_write,
      sep = "\t",
      row.names = F,
      col.names = T,
      quote = FALSE
    )
    write.table(
      test_data,
      test_write,
      sep = "\t",
      row.names = F,
      col.names = T,
      quote = FALSE
    )
  
  
}
  

  