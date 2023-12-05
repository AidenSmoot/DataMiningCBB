import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from numpy import log,dot,exp,shape
from sklearn.metrics import r2_score 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

#read in the preprocessed data and drop unnecessary columns
cbb = pd.read_csv('Basketball Data Mining/finalCBB.csv')
cbb = cbb.drop(cbb.columns[0:5],axis=1)
cbb = cbb.drop(['YEAR'],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)

#read in a new row of data
newData = [52.6, 47.7, 14, 22.8, 30, 29.7, 28.4, 23.9]
predictionsByYear = []
weightsByYear = []

#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb.csv')
cbbOriginal = cbbOriginal.drop(['YEAR'],axis=1)
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

#standardize the new data
newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

#define the logistic regression as a class with functions for each step
class LogisticRegression:
    #use the sigmoid function to predict values
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    #add a bias column of ones to the data and create a vector of weights (set to 0) 
    def initialize(self,X):
        try:
            weights = np.zeros((shape(X)[1]+1,1))
            X = np.c_[np.ones((shape(X)[0],1)),X]
        except:
            weights = np.zeros(9)
            X = np.insert(X, 0, 1.0)
        return weights,X
    #perform gradient descent to obtain the optimal parameters
    def fit(self,X,y,alpha=0.001,iter=1000):
        weights,X = self.initialize(X)
        #compute the cost function using log likelihood
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost[0]
        cost_list = np.zeros(iter,)
        #update the weights
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[int(i)] = cost(weights)
        self.weights = weights
        return cost_list
    #predict the win rate 
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
           lis.append(i)
        return lis

obj1 = LogisticRegression()
numFolds = 20
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
#perform 20-fold cross validation 
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    #divide up the data into training and testing sets
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    #fit the model on the training data and predict on the test data
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    #compute the RMSE for each fold
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
  #keep track of the minimum RMSE (best model)
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0

#keep track of the start and end times to run the algorithm
start_time = time.time()
preds = obj1.predict(inputs)
end_time = time.time()

#compute the run time of the algorithm 
runTime = end_time - start_time

#store the predicted win rates
predicted = []
for i in range(len(preds)):
    predicted.append(preds[i][0])

#store the actual win rates
actual = []
actualValues = np.ravel(output)
for i in range(len(actualValues)):
    actual.append(actualValues[i])


n = len(inputs.axes[0])
p = len(inputs.axes[1])
#Compute the R^2 value
R2 = 1-(np.min(rmses**2)/np.var(output)) # R^2: 1 - (residual sum of squares / total sum of squares)
#compute the adjusted R^2 value
adj_R2 = 1- (1 - R2) * ((n-1)/(n-p-1))

# Calculate the likelihood for each observation
likelihoods = []
for i in range(len(actual)):
    likelihoods.append(actual[i] * np.log(predicted[i]) + (1 - actual[i]) * np.log(1 - predicted[i]))

# The negative log-likelihood is the negation of the sum of the log-likelihoods
nll = -np.sum(likelihoods)

#calculate the AIC
aic = 2 * p - 2 * nll

#calculate the BIC
bic = np.log(n) * p - 2 * nll

# Add a constant to the features (required for statsmodels)
features_with_constant = add_constant(inputs)

#Create a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data["Variable"] = features_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(features_with_constant.values, i) for i in range(features_with_constant.shape[1])]


#store the weights in a list
reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

#add the bias term (intercept) to the list 
attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])

columnWeights = pd.DataFrame({"Attribute": attributes, "Weights": reformattedWeights})

overallPrediction = obj1.predict(newDataStd)


#run the model on each year
numFolds = 5
weightsOverTime = []
for i in range(len(attributes)):
    weightsOverTime.append([])

cbb = pd.read_csv('Basketball Data Mining/finalCBB13.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb13.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])

for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)



cbb = pd.read_csv('Basketball Data Mining/finalCBB14.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)

#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb14.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)



cbb = pd.read_csv('Basketball Data Mining/finalCBB15.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb15.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)


cbb = pd.read_csv('Basketball Data Mining/finalCBB16.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb16.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)


cbb = pd.read_csv('Basketball Data Mining/finalCBB17.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb17.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)



cbb = pd.read_csv('Basketball Data Mining/finalCBB18.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb18.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])


predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)




cbb = pd.read_csv('Basketball Data Mining/finalCBB19.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb19.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)



cbb = pd.read_csv('Basketball Data Mining/finalCBB20.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb20.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])


predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)


cbb = pd.read_csv('Basketball Data Mining/finalCBB21.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb21.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])


predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)


cbb = pd.read_csv('Basketball Data Mining/finalCBB22.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb22.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0

reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])


for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])



predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)

cbb = pd.read_csv('Basketball Data Mining/finalCBB23.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)


#read in the original, non-normalized data
cbbOriginal = pd.read_csv('Basketball Data Mining/cbb23.csv')
cbbOriginal = cbbOriginal.drop(['WR'], axis=1)

#compute the mean and standard deviation of each column
means = [] 
stdevs = []
for column in cbbOriginal:
    means.append(np.mean(cbbOriginal[column]))
    stdevs.append(np.std(cbbOriginal[column]))

newDataStd = []
for i in range(len(newData)):
    newDataStd.append((newData[i] - means[i]) / stdevs[i])

obj1 = LogisticRegression()
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
minRMSE = 10
bestWeights = np.zeros(9)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    obj1.fit(inputsTrain, outputTrain)
    preds = obj1.predict(inputsTest)
    res = np.zeros(outputTest.shape[0])
    tot = np.zeros(outputTest.shape[0])
    mean = np.mean(outputTest, axis = 0)
    for j in range(inputsTest.shape[0]):
        res[j] = (outputTest.iloc[j]-preds[j])**2 # Residual sum of squares: (output - predicted)^2
        tot[j] = (outputTest.iloc[j]-mean)**2 # Total sum of squares: (output - mean)^2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])) # Root mean squared error: sqrt(sum(residuals)/variance(output))
    print("RMSE = ", rmses[i])
    if rmses[i] < minRMSE:
        minRMSE = rmses[i]
        bestWeights = np.ravel(obj1.weights)
    increment = 0


reformattedWeights = []
for i in range(0, len(bestWeights)):
    reformattedWeights.append(bestWeights[i])

attributes = ["BIAS"]
for i in range(0, len(inputs.columns)):
    attributes.append(inputs.columns[i])

predictionsByYear.append(obj1.predict(newDataStd))
weightsByYear.append(reformattedWeights)

for i in range(0, len(attributes)):
    weightsOverTime[i].append(reformattedWeights[i])

years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


for i in range(1, len(attributes)):
    plt.plot(years, weightsOverTime[i], label = attributes[i])
plt.legend()
plt.xlabel("Year")
plt.ylabel("Weight")
plt.show()


plt.plot(years, predictionsByYear)
plt.show()

print(columnWeights)
print("CBB Original: ")
print(cbbOriginal[1:10])
print('Means:')
print(means)
print('Standard Deviations: ')
print(stdevs)
print('weights: ')
print(weightsByYear)

