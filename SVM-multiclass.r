
SkeletonSize <- read.csv("C:/Users/Koft/Desktop/Dropbox/SkeletonSize.csv")
View(SkeletonSize)
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

is.data.frame(total)
typeof(total)
finaldf<-as.data.frame(total)

#SVM from ksvm {kernlab} (Με γραμμικό πυρήνα έχω 6 λάθη, ενώ με Gaussian Kernel ή πολυωνυμικό έχω 11)


library("kernlab", lib.loc="C:/Program Files (x86)/R/R-3.1.1/library")

	#Shuffle data
finaldf <- finaldf[sample(1:nrow(finaldf),length(1:nrow(finaldf))),1:ncol(finaldf)]
	
	#create train and test set 60-40
library("caTools", lib.loc="C:/Program Files (x86)/R/R-3.1.1/library")

Y = finaldf[,15] # extract labels from the data
msk = sample.split(Y, SplitRatio=6/10)
finaltrain = finaldf[ msk,1:15]  # use output of sample.split to ...
finaltest  = finaldf[!msk,1:15] 


## train a support vector machine
identification <- ksvm(label~.,data=finaltrain,kernel="rbfdot",
                       kpar=list(sigma=0.05),C=45,cross=10)
identification

## predict skeleton type on the test set
skeletonid<- predict(identification,finaltest[,-15])

A<-table(skeletonid,finaltest[,15])

library(caret)

confusionMatrix(A)
proc.time() - ptm


library(pryr)
mem_used()