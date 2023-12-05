RMSEs = read.csv('/Users/willowarana/Desktop/Data Mining/Project/R Code/ComparingRMSES.csv')
RMSEs <- subset(RMSEs, select = -c(5,6,7))

#store the alpha value and number of folds 
alpha = 0.05
k = 20

#---------------Comparison of OLS and SGD-----------------

#create a list with the error rate differences for each fold in OLS and SGD
d1 <- RMSEs$LR - RMSEs$SGD

#calculate the average difference in error rates
d1_avg <- mean(d1)

#calculate the variance in differences
sumDiff <- 0
for (x in 1:k) {
  sumDiff <- sumDiff + ((d1[x] - d1_avg)^2)
}
  
d1_var <- sumDiff / (k * (k-1)) 

#calculate the confidence interval
upper <- d1_avg + (qt((1-alpha)/2,k-1) * sqrt(d1_var))
lower <- d1_avg - (qt((1-alpha)/2,2) * sqrt(d1_var))

#print the output
print(paste('The confidence level is ', 1- alpha, '%'))
print(paste('The confidence interval is [', lower, ',', upper, ']'))
if (upper / lower > 0) {
  print('The error rate difference is significant.')
  if(d1_avg < 0) {
    print('The LR model is the selected model.') 
  } else {
    print('The SGD model is the selected model.') }
} else {
  print('The error rate difference is not significant. Either model can be selected.') }

#---------------Comparison of OLS and ANN-----------------

#create a list with the error rate differences for each fold in OLS and ANN
d2 <- RMSEs$LR - RMSEs$ANN

#calculate the average difference in error rates
d2_avg <- mean(d2)

#calculate the variance in differences
sumDiff <- 0 
for (x in 1:k){
  sumDiff <- sumDiff + ((d2[x] - d2_avg)^2) 
  }

d2_var <- sumDiff / (k * (k-1)) 

#calculate the confidence interval
upper <- d2_avg + (qt((1-alpha)/2,k-1) * sqrt(d2_var))
lower <- d2_avg - (qt((1-alpha)/2,2) * sqrt(d2_var))

#print the output
print(paste('The confidence level is ', 1- alpha, '%'))
print(paste('The confidence interval is [', lower, ',', upper, ']'))
if (upper / lower > 0) {
  print('The error rate difference is significant.')
  if(d2_avg < 0) {
    print('The LR model is the selected model.') 
  } else {
    print('The ANN model is the selected model.') }
} else {
  print('The error rate difference is not significant. Either model can be selected.') }


#---------------Comparison of OLS and KNN-----------------

#create a list with the error rate differences for each fold in OLS and KNN
d3 <- RMSEs$LR - RMSEs$KNN

#calculate the average difference in error rates
d3_avg <- mean(d3)

#calculate the variance in differences
sumDiff <- 0 
for (x in 1:k){
  sumDiff <- sumDiff + ((d3[x] - d3_avg)^2) 
}

d3_var <- sumDiff / (k * (k-1)) 

#calculate the confidence interval
upper <- d3_avg + (qt((1-alpha)/2,k-1) * sqrt(d3_var))
lower <- d3_avg - (qt((1-alpha)/2,2) * sqrt(d3_var))

#print the output
print(paste('The confidence level is ', 1- alpha, '%'))
print(paste('The confidence interval is [', lower, ',', upper, ']'))
if (upper / lower > 0) {
  print('The error rate difference is significant.')
  if(d3_avg < 0) {
    print('The LR model is the selected model.') 
  } else {
    print('The KNN model is the selected model.') }
} else {
  print('The error rate difference is not significant. Either model can be selected.') }

#---------------Comparison of SGD and KNN-----------------

#create a list with the error rate differences for each fold in SGD and KNN
d4 <- RMSEs$SGD - RMSEs$KNN

#calculate the average difference in error rates
d4_avg <- mean(d4)

#calculate the variance in differences
sumDiff <- 0 
for (x in 1:k){
  sumDiff <- sumDiff + ((d4[x] - d4_avg)^2) 
}

d4_var <- sumDiff / (k * (k-1)) 

#calculate the confidence interval
upper <- d4_avg + (qt((1-alpha)/2,k-1) * sqrt(d4_var))
lower <- d4_avg - (qt((1-alpha)/2,2) * sqrt(d4_var))

#print the output
print(paste('The confidence level is ', 1- alpha, '%'))
print(paste('The confidence interval is [', lower, ',', upper, ']'))
if (upper / lower > 0) {
  print('The error rate difference is significant.')
  if(d4_avg < 0) {
    print('The SGD model is the selected model.') 
  } else {
    print('The KNN model is the selected model.') }
} else {
  print('The error rate difference is not significant. Either model can be selected.') }

#---------------Comparison of SGD and ANN-----------------

#create a list with the error rate differences for each fold in SGD and ANN
d5 <- RMSEs$SGD - RMSEs$ANN

#calculate the average difference in error rates
d5_avg <- mean(d5)

#calculate the variance in differences
sumDiff <- 0 
for (x in 1:k){
  sumDiff <- sumDiff + ((d5[x] - d5_avg)^2) 
}

d5_var <- sumDiff / (k * (k-1)) 

#calculate the confidence interval
upper <- d5_avg + (qt((1-alpha)/2,k-1) * sqrt(d5_var))
lower <- d5_avg - (qt((1-alpha)/2,2) * sqrt(d5_var))

#print the output
print(paste('The confidence level is ', 1- alpha, '%'))
print(paste('The confidence interval is [', lower, ',', upper, ']'))
if (upper / lower > 0) {
  print('The error rate difference is significant.')
  if(d5_avg < 0) {
    print('The SGD model is the selected model.') 
  } else {
    print('The ANN model is the selected model.') }
} else {
  print('The error rate difference is not significant. Either model can be selected.') }



