library(DMwR)
#require(xgboost)
require(methods)

#setwd("D:\AA_YN\RH")
setwd("D:/AA_YN/BDXH/SMOTE2/SMOTE_NET")
data_train = read.csv("Elastic_net2_0.03.csv",header = F)
data_train$V1=factor(data_train$V1)
train_data_SMOTEdata <- SMOTE(V1~.,data_train,perc.over =403.14,perc.under=124.8)
jishu<-table(train_data_SMOTEdata$V1)


write.csv(train_data_SMOTEdata,file='SMOTE_NET_0.03.csv')


#data_test<- read.csv("test_data.csv",header = F)
#data_test$V1=factor(data_test$V1)
#test_SMOTEdata <- SMOTE(V1~.,data_test,perc.over =549.54,perc.under=118.2)
#test_jishu<-table(test_SMOTEdata$V1)
#write.csv(test_SMOTEdata,file='test_SMOTEdata.csv')


