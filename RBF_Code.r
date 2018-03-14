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


#Create new dataframe to apply RBF
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


#Apply RBF model
model<-rbf(total$inputsTrain, total$targetsTrain, size = c(15), maxit = 80,
  initFunc = "RBF_Weights", initFuncParams = c(0, 1, 0, 0.02, 0.04),
  learnFunc = "RadialBasisLearning", learnFuncParams = c(1e-05, 0, 1e-05,
  0.1, 0.8), shufflePatterns = TRUE, linOut = FALSE, inputsTest = total$inputsTest,
  targetsTest = total$targetsTest)

## Apply RBF model with DDA:
#model<-rbfDDA(total$inputsTrain, total$targetsTrain, maxit = 15, initFunc = "Randomize_Weights",
  #initFuncParams = c(-0.3, 0.3), learnFunc = "RBF-DDA",
  #learnFuncParams = c(0.4, 0.2, 5), shufflePatterns = TRUE, linOut = FALSE)

#Prints out a summary of the network
summary(model)
model

#Extract the weight matrix
#weightMatrix(model)

#Create a a list containing information extracted from the network
#extractNetInfo(model)

#Create Plots to test overtraining 
par(mfrow=c(2,2))
plotIterativeError(model)
predictions <- predict(model,total$inputsTest)

#ÂCreate Confusion Matrix and calculate Model's statistics
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




