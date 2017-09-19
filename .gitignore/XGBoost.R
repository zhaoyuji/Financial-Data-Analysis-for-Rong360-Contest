---
  title: "xgboost"
output: html_document
---
  

# load package
library(xgboost)
library(caret)
library(AUC)
library(dplyr)

# load data
order <- read.table("order_train.txt",sep="\t",header=TRUE)
product <- read.table("product.final.txt",sep="\t",header=TRUE)
user <- read.table("user.final.txt",sep="\t",header=TRUE)
quality <- read.table("quality.final.txt",sep="\t",header=TRUE)

# delete duplicated data
user <- base::unique(user)
quality <- plyr::count(quality)

# rename
quality$term <- quality$application_term
quality$limit <- quality$application_limit
quality <- subset(quality, select = c(-application_term,-application_limit))

# retain the most recent data
user <-  user[order(user$user_id, -user$date),]
user <-  user[!duplicated(user$user_id),]
user <- subset(user, select = -date)

# merge data
dataset <- merge(order, user, by = "user_id", all.x = T)
dataset <- merge(dataset, product, by = "product_id", all.x = T)
dataset <- merge(dataset, quality, by = c("user_id","limit","term","standard_type","guarantee_type"), all.x = T)


# delete misssing value
delete <- apply(dataset,1,function(x) sum(is.na(x)))
dataset <- dataset[which(delete<ncol(dataset)*0.5),]
delete <- apply(dataset,2,function(x) sum(is.na(x)))
dataset <- dataset[,which(delete<nrow(dataset)*0.3)]

# delete useless variables
dataset <- subset(dataset, select = -c(bank_id.x,bank_id.y,
                                       city_id.x,city_id.y, 
                                       user_id, product_id,
                                       product_type.y,freq))

# factor variables
dataset$date <- as.factor(dataset$date%%7)

factor <- subset(dataset,select = c(result, application_type, bank,
                                    col_type,
                                    tax,guarantee_required,
                                    loan_term_type,
                                    business_license,car,guarantee_type,
                                    house,house_register,id,income,
                                    interest_rate_type,qid77,
                                    lifecost,married,mobile_verify,op_type,
                                    platform,product_type.x,quality,
                                    repayment_type,socialsecurity
                                    ,user_loan_experience,is_paid))

factor <- mutate_if(factor, is.integer, as.factor)

dataset <- subset(dataset,select = -c(result, application_type, bank,
                                      col_type,
                                      tax,guarantee_required,
                                      loan_term_type,
                                      business_license,car,guarantee_type,
                                      house,house_register,id,income,
                                      interest_rate_type,qid77,
                                      lifecost,married,mobile_verify,op_type,
                                      platform,product_type.x,quality,
                                      repayment_type,socialsecurity
                                      ,user_loan_experience,is_paid))

dataset <- cbind(factor, dataset)

# delete low variance
nzv = nearZeroVar(dataset)
dataset = subset(dataset, select = -nzv)

# partition the data
index <- createDataPartition(dataset$result, p = .7, list = FALSE)
trainset <- dataset[index,]
testset <- dataset[-index,]


# ratio of positive sample and negative sample
sumneg <- sum(trainset$result==0)
sumpos <- sum(trainset$result==1)

# change dataframe as matrix
result <- trainset$result
trainset <- subset(trainset, select = -result)
train <- data.matrix(trainset)

testresult <- testset$result
testresult <- as.factor(testresult)
testset <- subset(testset, select = -result)
test <- data.matrix(testset)

# xgboost model
result <- as.numeric(result)-1

param <- list(max_depth = 25, eta = 0.1,
              objective = "binary:logistic", 
              eval_metric = "auc",
              scale_pos_weight = sumneg / sumpos,
              silent = 1,
              nthrea= 16,
              max_delta_step=4,
              subsample=0.8,min_child_weigth=2)
set.seed(13)
XG <- xgboost(data = train, label = result, params =param, nrounds = 100)

# AUC
pred <- predict(XG,test)
roc <- roc(pred, testresult)
plot(roc$fpr, roc$tpr, type = "l", col = "blue")
auc(roc(pred, testresult))

# F1-score
Fscore <- function(threshold){
  TP <- length(which(pred >= threshold & testresult==1))
  TN <- length(which(pred < threshold & testresult==0))
  F <- 2*TP/(nrow(testset)+TP-TN)
  return(F)
}
threshold <- seq(from=0, to=1, by=0.01)
f1_score <- sapply(threshold, function(x) Fscore(x))
plot(threshold, f1_score)
max(f1_score)

# confusion matrix
confusionMatrix(ifelse(pred > threshold[which(f1_score==max(f1_score))],1,0), testresult)

# importance variables
importance <- xgb.importance(colnames(train), model = XG)  
xgb.plot.importance(importance[1:30], measure = 'Gain', xlab = "Gain")


