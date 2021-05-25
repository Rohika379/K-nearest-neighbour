#Packages required

install.packages('caTools')  #for train and test data split
install.packages('dplyr')    #for Data Manipulation
install.packages('ggplot2')  #for Data Visualization
install.packages('class')    #KNN 
install.packages('caret')    #Confusion Matrix
install.packages('corrplot') #Correlation Plot

library(caTools)
library(dplyr)
library(ggplot2)
library(caret)
library(class)
library(corrplot)

#Dataset 

glass <- read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\KNN\\glass.csv")

standard.features <- scale(glass[,1:9])

#Join the standardized data with the target column
data <- cbind(standard.features,glass[10])
#Check if there are any missing values to impute. 
anyNA(data)

#Train test split

set.seed(101)

sample <- sample.split(data$Type,SplitRatio = 0.70)

train <- subset(data,sample==TRUE)

test <- subset(data,sample==FALSE)

#Taking type as our output variable

predicted.type <- knn(train[1:9],test[1:9],train$Type,k=1)
#Error in prediction
error <- mean(predicted.type!=test$Type)
error

predicted.type <- NULL
error.rate <- NULL

for (i in 1:10) {
  predicted.type <- knn(train[1:9],test[1:9],train$Type,k=i)
  error.rate[i] <- mean(predicted.type!=test$Type)
  
}

knn.error <- as.data.frame(cbind(k=1:10,error.type =error.rate))

ggplot(knn.error,aes(k,error.type))+ 
  geom_point()+ 
  geom_line() + 
  scale_x_continuous(breaks=1:10)+ 
  theme_bw() +
  xlab("Value of K") +
  ylab('Error')

predicted.type <- knn(train[1:9],test[1:9],train$Type,k=5)
#Error in prediction
error <- mean(predicted.type!=test$Type)
error

