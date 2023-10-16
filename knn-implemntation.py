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
import math as math

# BAscketball data
data = np.loadtxt('finalCBB.csv', delimiter=',', skiprows=1,usecols=(5,6,7,8,9,10, 11,12,13) )
[n,p] = np.shape(data)

# percentages used for training and testing respectively
num_train = int(0.75*n)
num_test = n - num_train

# split data into training set and testing set   
sample_train = data[0:num_train,0:8]
sample_test = data[num_train:,0:8]
label_train = data[0:num_train,8]
label_test = data[num_train:,8]

#See were is the lower error in data
p_error = []
c_error = []
train_error = []
k_values = range(1,50)
#k_values = range(3,7)
#k_values = range(5,6)
num = np.shape(label_test)
num_t = np.arange(num[0])
#plt.subplots(2,2)
#plt.subplots(1,1)
i = 1
error_test = 0
max_error = 0 
max_k = 0
for k in k_values:
        model = neighbors.KNeighborsRegressor(k, weights= "distance")
        predicted_label = model.fit(sample_train, label_train).predict(sample_test)
        #error_test = mean_squared_error(label_test, predicted_label)
        error_test = model.score(sample_test, label_test)
        #error_train = model.score(sample_test, label_test)
        p_error.append(error_test)
        if ( max_error < error_test ): 
            max_error = error_test
            max_k = k 
        #train_error.append(error_train)
        #plt.subplot(2,2,i)
        #plt.plot(num_t,predicted_label,color ="red", label="prediction with k = %s" % (k), linewidth= 0.2)
        #plt.plot(num_t,label_test,color="blue", label="Actual data", linewidth= 0.2)
        #plt.xlabel("Records")
        #plt.ylabel("Win-Rate")
        #if ( i == 1 or i== 3):
            # plt.ylabel("Win-Rate")
        #if( i == 4 or i== 3):
            #plt.xlabel("Records")
        #plt.axis("tight")
        #i = i + 1

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
        #c_error.append(error)
        #plt.subplot(1,1,i)
        #plt.plot(error,color ="red", label="Cross-val, k = %s" % (k), linewidth= 0.2)
        #plt.axis("tight")
        #if ( i == 1 or i== 3):
            #plt.ylabel("CV error")
        #if( i == 4 or i== 3):
            #plt.xlabel(" cv values 1-20")  
        #plt.ylabel("CV error")
        #plt.xlabel(" cv values 1-20")
        #plt.legend()
        #plt.title("Cross-validation k = %s, cv = 20"%(k))
        #i = i + 1
      
plt.show()

#plt.plot(k_values,p_error, marker = 'o', markersize = 2.0)
#plt.ylabel("Mean square errror")
#plt.xlabel("Value of k")
#plt.legend()
#plt.title("KNN MSE  with  distace as weight")

plt.plot(k_values,p_error, marker = 'o', markersize = 2.0)
plt.ylabel("R^2  squared ")
plt.xlabel("K value ")
plt.legend()
plt.title("KNN R^2")

#plt.plot(k_values,c_error, marker = 'o', markersize = 1.0
#plt.legend()
#plt.ylabel("Cross-validate-error")
#plt.xlabel("Value of k")
#plt.title("KNN CROSS-VALIDATION")

#Calculation of the VIF for each feature

print("Calculation of the BIC: ")
bic = - 2* math.log2(max_error) + math.log2(num_train) * p 
print(bic)


print( "The maximum error is : ")
print( max_error)

print( "The maximum k is : ")
print( max_k)
    
#Calculate the AIC for the best model 
print ("Computation of AIC")
aic =  - 2 * math.log2(max_error) + 2 * p 
print(aic)



    
        
    


