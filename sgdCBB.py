from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
splitPoint = rowCount // 5
Xtrain, Xtest, ytrain, ytest = inputs.iloc[splitPoint:], inputs.iloc[:splitPoint],output.iloc[splitPoint:], output.iloc[:splitPoint]
rgr.fit(Xtrain,np.ravel(ytrain))
preds = cross_val_predict(rgr,Xtest,np.ravel(ytest), cv=20)
res = np.zeros(ytest.shape[0])
tot = np.zeros(ytest.shape[0])
mean = np.mean(ytest, axis = 0)
for i in range(Xtest.shape[0]):
    res[i] = (ytest.iloc[i]-preds[i])**2
    tot[i] = (ytest.iloc[i]-mean)**2
    col = 'r' if res[i] > .01 else 'b'
    plt.scatter(i,res[i], color=col)
plt.ylabel("Residuals")
plt.xlabel("Test Cases")
print("R^2 = ", 1-(np.sum(res)/np.sum(tot)))
print("MSE = ", (np.sum(res)/ytest.shape[0])**(1/2))
plt.show()

