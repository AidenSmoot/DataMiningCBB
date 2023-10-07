from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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


numIndVar = len(inputs.columns)
colors = ['tab:red','tab:green','tab:blue','tab:pink','tab:purple','tab:cyan','tab:olive','tab:brown']
plt.subplots(4,2)
for i in range(numIndVar):
    colName = inputs.columns[i]
    rgr.fit(inputs[[colName]],output['WR'])
    preds = cross_val_predict(rgr,inputs[[colName]],output['WR'], cv=20)
    plt.subplot(4,2,i+1)
    plt.plot(preds,output['WR'],color = 'b',label="Predicted " + colName)
    # plt.plot(inputs[[colName]],output['WR'],color = 'r',alpha = .2, label="Real " + colName)
    plt.plot([0,1],[0,1],color='k', linestyle='dashed',label = 'Perfect R^2')
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.legend()
    print(colName)
    preds = pd.DataFrame(np.reshape(preds,(preds.shape[0],1)),columns=[colName])
    # print("Slope = ", rgr.coef_)
    # print("Intercept = ", rgr.intercept_)
    print("Real R-squared = ", rgr.score(inputs[[colName]],output['WR']))
    print("Pred R-squared = ", rgr.score(preds,output['WR']))
    print("MSE = ", mean_squared_error(output['WR'], preds))
    print('-----------------------')
plt.show()

