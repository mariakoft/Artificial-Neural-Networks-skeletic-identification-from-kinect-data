ptm <- proc.time()

#Create ThoracicSpine and Height attributs

SkeletonSize$ThoracicSpine<- SkeletonSize$ShoulderCenterToSpine + SkeletonSize$SpineToHipCenter
SkeletonSize$legssum<-SkeletonSize$RightLowerLeg+SkeletonSize$LeftLowerLeg+SkeletonSize$LeftUpperLeg+ SkeletonSize$RightUpperLeg
SkeletonSize$legmean<-SkeletonSize$legssum /2
SkeletonSize$Height<- SkeletonSize$ThoracicSpine + SkeletonSize$legmean +SkeletonSize$Neck


#Delete entries (variables)

SkeletonSizen<-SkeletonSize[-(1:7)]
View(SkeletonSizen)
SkeletonSizene<-SkeletonSizen[-(6)]
View(SkeletonSizene)
SkeletonSizenes<-SkeletonSizene[-(8:10)]
View(SkeletonSizenes)
SkeletonSizeness<-SkeletonSizenes[-(12:13)]
View(SkeletonSizeness)
SkeletonSizenesse<-SkeletonSizeness[-(13)]
View(SkeletonSizenesse)
SkeletonSizeFinal<-SkeletonSizenesse[-(14:15)]
View(SkeletonSizeFinal)

#Create and label sub-datasets

IIPerson1data<-SkeletonSizeFinal[1:35,]
IPerson2data<-SkeletonSizeFinal[36:85,]
IPerson4data<-SkeletonSizeFinal[86:144,]
IPerson5data<-SkeletonSizeFinal[145:190,]
IPerson3data<-SkeletonSizeFinal[225:276,]

IIPerson1data$label<-c("10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000","10000","10000", "10000","10000","10000")
IPerson2data$label<-c("01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000", "01000","01000", "01000", "01000", "01000")
IPerson4data$label<-c("00010","00010", "00010", "00010", "00010", "00010","00010", "00010","00010","00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010", "00010","00010", "00010", "00010", "00010")
IPerson3data$label<-c("00100","00100","00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100", "00100","00100", "00100", "00100", "00100")
IPerson5data$label<-c("00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001", "00001","00001", "00001", "00001", "00001","00001")


#Create new Dataframe to apply MLP
total<-rbind(IPerson2data, IIPerson1data)
total<-rbind(total,IPerson4data)
total<-rbind(total, IPerson5data)
total<-rbind(total,IPerson3data)

#Prepare train and test set (60-40)
	#Shuffle data
total <- total[sample(1:nrow(total),length(1:nrow(total))),1:ncol(total)]
library(RSNNS)
 
totalValues <- total[,1:14]
totalTargets <- decodeClassLabels(total[,15])
#totalTargets <- decodeClassLabels(total[,15], valTrue=0.9, valFalse=0.1)

	#Split Dataset (Ratio 60-40)
total <- splitForTrainingAndTest(totalValues, totalTargets, ratio=0.40)

	#Normalize test and train set
total <- normTrainingAndTestSet(total)


#Apply MPL model
model<-mlp(total$inputsTrain, total$targetsTrain, size=5, maxit = 13,
  initFunc = "Randomize_Weights", initFuncParams = c(-0.1, 0.1),
  learnFunc = "Std_Backpropagation", learnFuncParams = c(0.05, 0.1),
  hiddenActFunc = "Act_Signum", shufflePatterns = TRUE, linOut = FALSE,
  inputsTest=total$inputsTest, targetsTest=total$targetsTest)


#Prints out a summary of the network
summary(model)
model

#Extract the weight matrix
weightMatrix(model)

#Create a a list containing information extracted from the network
extractNetInfo(model)

#Create Plots to test overtraining and Regression error
par(mfrow=c(2,2))
plotIterativeError(model)
predictions <- predict(model,total$inputsTest)
plotRegressionError(predictions[,2], total$targetsTest[,2])

#Create Confusion Matrix and calculate Model's statistics
confusionMatrix(total$targetsTrain,fitted.values(model))
confusionMatrix(total$targetsTest,predictions)

F<-confusionMatrix(total$targetsTrain,fitted.values(model))
D<-as.matrix(F)
K<-confusionMatrix(total$targetsTest,predictions)
G<-as.matrix(K)

library(caret)

confusionMatrix(D)
confusionMatrix(G)

#Training time and memory use

proc.time() - ptm

library(pryr)
mem_used()




