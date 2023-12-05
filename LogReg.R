#load the data and packages
regData <- read.csv('finalCBB.csv')
library(caret)

#remove the categorical variables
regData <- subset(regData, select = -c(1,11))

#update a data point that is incorrect
regData[2796, 'WR'] <- 20/24

#perform 20-fold cross-validation logistic regression
twentyFoldCV <- trainControl(method = 'cv', number = 20)
cv_model <- train(WR ~ ., data = regData, method = 'glm', family = "binomial", trControl = twentyFoldCV)

#output the RMSEs
rmses <- cv_model$resample$RMSE
write.csv(rmses, 'LR_rmses')
