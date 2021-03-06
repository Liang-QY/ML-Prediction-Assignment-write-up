#import
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(e1071)
library(randomForest)
install.packages('rattle')
set.seed(1) 
#download
train_raw = read.csv("/Users/apple/Downloads/pml-training.csv")
test_raw= read.csv("/Users/apple/Downloads/pml-testing.csv") 
#replace with NA
train<-read.csv("/Users/apple/Downloads/pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test<-read.csv("/Users/apple/Downloads/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
#clean
install.packages('dplyr')
library(dplyr)
train_clean<-train[,colSums(is.na(train))==0] 
test_clean<-test[,colSums(is.na(test))==0]
train_clean<-train_clean[,-(0:7)]
test_clean<-test_clean[,-(0:7)]
# Check for near zero variance predictors and drop them if necessary
#nzv <- nearZeroVar(train.data.clean1,saveMetrics=TRUE)
#zero.var.ind <- sum(nzv$nzv)

#if ((zero.var.ind>0)) {
#train.data.clean1 <- train.data.clean1[,nzv$nzv==FALSE]
#}
set.seed(3)
train1<-createDataPartition(train_clean$classe,p=0.75,list = F)
training<-train_clean[train1,]
crossval<-train_clean[-train1,]

#model
set.seed(1)
control<-trainControl(method='cv',5)
mod_rf<-train(classe~.,data=training,method="rf",trcontrol=control)
mod_rf

#in-train
trainpredict<-predict(mod_rf,training)
confusionMatrix(training$classe, trainpredict)
#out-of-sample
rf.predict<-predict(mod_rf,crossval)
confusionMatrix(crossval$classe, rf.predict)

#test
testpredict<-predict(mod_rf, test)
testpredict
