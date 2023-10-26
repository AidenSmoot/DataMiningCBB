from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

cbb = pd.read_csv('Basketball Data Mining\\finalCBB.csv')
cbb = cbb.drop(cbb.columns[0:5],axis=1)
cbb = cbb.drop(['YEAR'],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)
rgr = SGDRegressor(loss='squared_error', penalty='l2', max_iter=10000)

'''
Use train test split
Run fit with training data
Run cross_val_predict with test data
'''

numIndVar = len(inputs.columns)
# splitPoint = rowCount // 5
# Xtrain, Xtest, ytrain, ytest = inputs.iloc[splitPoint:], inputs.iloc[:splitPoint],output.iloc[splitPoint:], output.iloc[:splitPoint]
# rgr.fit(Xtrain,np.ravel(ytrain))
# preds = cross_val_predict(rgr,Xtest,np.ravel(ytest), cv=20)
# res = np.zeros(ytest.shape[0])
# tot = np.zeros(ytest.shape[0])
# mean = np.mean(ytest, axis = 0)
# for i in range(Xtest.shape[0]):
#     res[i] = (ytest.iloc[i]-preds[i])**2
#     tot[i] = (ytest.iloc[i]-mean)**2
#     col = 'r' if res[i] > .01 else 'b'
#     plt.scatter(i,res[i], color=col)
# plt.ylabel("Residuals")
# plt.xlabel("Test Cases")
# print("R^2 = ", 1-(np.sum(res)/np.sum(tot)))
# print("RMSE = ", np.sqrt((np.sum(res)/ytest.shape[0])**(1/2)))
# plt.show()


numFolds = 20
testSize = rowCount // numFolds
increment = 0
rmses = np.zeros(numFolds)
for i in range(numFolds):
    if (i == numFolds - 1):
        increment = increment + 2
    inputsTest = inputs.iloc[i*testSize:(i+1)*testSize+increment]
    inputsTrain = inputs.drop(inputs.index[i*testSize:(i+1)*testSize+increment])
    outputTest = output.iloc[i*testSize:(i+1)*testSize+increment]
    outputTrain = output.drop(output.index[i*testSize:(i+1)*testSize+increment])
    rgr.fit(inputsTrain,np.ravel(outputTrain))
    preds = rgr.predict(inputsTest)
    for j in range(inputsTest.shape[0]):
        res = np.zeros(outputTest.shape[0])
        tot = np.zeros(outputTest.shape[0])
        mean = np.mean(outputTest, axis = 0)
        res[j] = (outputTest.iloc[j]-preds[j])**2
        tot[j] = (outputTest.iloc[j]-mean)**2
    rmses[i] = np.sqrt((np.sum(res)/outputTest.shape[0])**(1/2))
    print("RMSE = ", rmses[i])
    increment = 0
print("R^2 = ", 1-(np.max(rmses**2)/np.var(output)))

