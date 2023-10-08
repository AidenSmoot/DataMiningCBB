# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 07:46:06 2023

@author: teres
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


# BAscketball data
data = np.loadtxt('finalCBB.csv', delimiter=',', skiprows=1,usecols=(3,4,5,6,7,8,9,10, 11,12,13) )
[n,p] = np.shape(data)

# percentages used for training and testing respectively
num_train = int(0.75*n)
num_test = n - num_train

# split data into training set and testing set   
sample_train = data[0:num_train,0:10]
sample_test = data[num_train:,0:10]
label_train = data[0:num_train,10]
label_test = data[num_train:,10]

#See were is the lower error in data
p_error = []
c_error = []
#k_values = range(1,50)
#k_values = range(3,7)
k_values = range(5,6)
num = np.shape(label_test)
num_t = np.arange(num[0])
#plt.subplots(2,2)
#plt.subplots(1,1)
i = 1

for k in k_values:
        model = neighbors.KNeighborsRegressor(k, weights= "distance")
        predicted_label = model.fit(sample_train, label_train).predict(sample_test)
        error_test = mean_squared_error(label_test, predicted_label)
        p_error.append(error_test)
        plt.subplot(2,2,i)
        plt.plot(num_t,predicted_label,color ="red", label="prediction with k = %s" % (k), linewidth= 0.2)
        plt.plot(num_t,label_test,color="blue", label="Actual data", linewidth= 0.2)
        #plt.xlabel("Records")
        #plt.ylabel("Win-Rate")
        if ( i == 1 or i== 3):
            plt.ylabel("Win-Rate")
        if( i == 4 or i== 3):
            plt.xlabel("Records")
        plt.axis("tight")
        i = i + 1

#plt.title("Comparison between predictons")
#plt.legend()
#plt.show()

i = 1
plt.subplots(1,1)
#plt.subplots(2,2)
for k in k_values:
        model = neighbors.KNeighborsRegressor(k, weights= "distance")
        predicted_label = model.fit(sample_train, label_train).predict(sample_test)
        error = cross_val_score(model, sample_test, label_test, cv = 20)
        c_error.append(error)
        plt.subplot(1,1,i)
        plt.plot(error,color ="red", label="Cross-val, k = %s" % (k), linewidth= 0.2)
        plt.axis("tight")
        #if ( i == 1 or i== 3):
            #plt.ylabel("CV error")
        #if( i == 4 or i== 3):
            #plt.xlabel(" cv values 1-20")  
        plt.ylabel("CV error")
        plt.xlabel(" cv values 1-20")
        plt.legend()
        plt.title("Cross-validation k = %s, cv = 20"%(k))
        #i = i + 1
      
plt.show()

#plt.plot(k_values,p_error, marker = 'o', markersize = 0.1)
#plt.ylabel("Mean square errror")
#plt.xlabel("Value of k")
#plt.legend()
#plt.title("KNN MSE  with  distace as weight")



#plt.plot(k_values,c_error, marker = 'o', markersize = 1.0)
#plt.legend()
#plt.ylabel("Cross-validate-error")
#plt.xlabel("Value of k")
#plt.title("KNN CROSS-VALIDATION")




    
        
    


