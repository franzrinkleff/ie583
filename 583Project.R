library(data.table)
library(Boruta)
library(RWeka) #j48
library(e1071) #nb
library(caret) # For CreatePartition and ConfusionMatrix
library(DMwR)
library(randomForest)

set.seed(1234)

numberRows = 1000
projDS <- fread('C:/Users//franz_000/Desktop/583/project/train_sample.csv', stringsAsFactors = F, nrows=numberRows)
summary(projDS)

colnames(projDS)[colnames(projDS) == 'is_attributed'] <- 'class'
str(projDS)

projDS$class <- as.factor(projDS$class)
levels(projDS$class) <- c("zeros", "ones")
head(projDS)

table(projDS$class)

#derived variables

#the hour minute and am_pm_ do not appear to be working.  
projDS$click_time <- as.POSIXct(as.character(projDS$click_time))
projDS <- projDS %>% 
  mutate(hour = hour(click_time),
    minute = minute(click_time),
    am_pm = ifelse(hour(click_time) >= 12, "AM", "PM"),
    ip_device_os_channel_app = paste(ip,'-',device,'-',os,'-',channel,'-',app),
    ip_device_os = paste(ip,'-',device,'-',os),
    ip_device = paste(ip,'-',device),
    ip_channel_app = paste(ip,'-',channel,'-',app),
    ip_channel = paste(ip,'-',channel),
    ip_app = paste(ip,'-',app),
    channel_app = paste(channel,'-',app),
    channel_app_os_device = paste(channel,'-',app,'-',os,'-',device),
    channel_os_device = paste(channel,'-',os,'-',device)
)    
head(projDS)

#factors
projDS$class <- factor(projDS$class)
projDS$am_pm <- factor(projDS$am_pm)
projDS$ip_device_os <- factor(projDS$ip_device_os)
projDS$ip_device <- factor(projDS$ip_device)
projDS$ip_channel_app <- factor(projDS$ip_channel_app)
projDS$ip_channel <- factor(projDS$ip_channel)
projDS$ip_app <- factor(projDS$ip_app)
projDS$channel_app <- factor(projDS$channel_app)
projDS$channel_app_os_device <- factor(projDS$channel_app_os_device)
projDS$channel_os_device <- factor(projDS$channel_os_device)

#cv
projTrain <- projDS

# seperate train and test 
#projIndex <- createDataPartition(projDS$class,p=.67,list=FALSE,times=1) 
#projTrain <- projDS[projIndex,] 
#projTest <- projDS[-projIndex,]
#head(projTrain)

#select attributes
Boruta.projTrain <- Boruta(class~., data=projTrain, doTrace=2)
Boruta.projTrain
attStats(Boruta.projTrain)
plot(Boruta.projTrain)

# we need to be sure to exclude attribted_time
#hour and minute and am_pm are reject now but they are all the same value.  need to fix that.  
#click time was rejected.  not suprisignly.  somewhat a "random value"
#the rest were suprinsingly confirm on on this limited train dataset

projTrainFS<-projTrain[, c(1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20)]
head(projTrainFS)

projTrainFS<-projTrain[, c(1,2,3,4,5,8)]
head(projTrainFS)

#cross validation with SMOTE J48
cvCount = 5
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="smote")
#ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="down")
treeGrid_dectree = expand.grid(C=(1:5)*0.1, M=(1:5))

folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))

train.accuracy.estimate = NULL
fold.accuracy.estimate = NULL

for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  
  #smote
  #trainingData2 = trainingData
  trainingData2 = SMOTE(class~., data=trainingData)
  #trainingData2 <-downSample(x=trainingData, y =trainingData$class)
  
  
  
  #trainingData2 = trainingData
  
  model_dectree = train(class~., data=trainingData2, method="J48", trControl=ctrl_cv, tuneGrid=treeGrid_dectree)
  best<-as.numeric(model_dectree$bestTune)
  train.accuracy.estimate[f] = as.numeric(model_dectree$results[best,3])
  fold.accuracy.estimate[f] = (table(predict(model_dectree,testData),testData$class)[1,1]+table(predict(model_dectree,testData),testData$class)[2,2])/length(testData$class)
}
mean(train.accuracy.estimate)
mean(fold.accuracy.estimate)

#random forest with cross validation
model<-randomForest(class~.,data=projTrain) 
varImpPlot(model)

cvCount = 5
ctrl_cv = trainControl(method = "cv", number = cvCount, savePred = T, classProb = T, sampling="smote")
cols = ncol(projTrainFS)
tunegrid <- expand.grid(.mtry=c(1:cols))
folds = split(sample(nrow(projTrainFS), nrow(projTrainFS),replace=FALSE), as.factor(1:cvCount))
train.accuracy.estimate = NULL
fold.accuracy.estimate = NULL
for(f in 1:cvCount){
  testData = projTrainFS[folds[[f]],]
  trainingData = projTrainFS[-folds[[f]],]
  trainingData2 = SMOTE(class~., data=trainingData)   
  RF_model <- train(class~., data=trainingData2, method="rf", tuneGrid=tunegrid, trControl=ctrl_cv)
  best<-as.numeric(RF_model$bestTune)
  show(RF_model)
  train.accuracy.estimate[f] = as.numeric(RF_model$results[best,3])
  fold.accuracy.estimate[f] = (table(predict(RF_model,testData),testData$class)[1,1]+table(predict(RF_model,testData),testData$class)[2,2])/length(testData$class)
}
mean(train.accuracy.estimate)
mean(fold.accuracy.estimate)

