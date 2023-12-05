#load the data and packages
regData <- read.csv('finalCBB.csv')
library(caret)

#remove the categorical variables
regData <- subset(regData, select = -c(1,2,3,4,5,15))

fiveFoldCV <- trainControl(method = 'cv', number = 20)

cv_model <- train(WR ~ ., data = regData, method = 'lm', trControl = fiveFoldCV)

summary(cv_model)

