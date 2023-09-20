# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:24:53 2023

@author: teres
"""
import numpy as np
import math

#files = ['cbb.csv', 'cbb13.csv', 'cbb14.csv' , 'cbb15.csv' , 'cbb16.csv', 'cbb17.csv',
         #'cbb18.csv', 'cbb19.csv', 'cbb21.csv', 'cbb22.csv', 'cbb23.csv']

# excluded season 20 - 
files = ['cbb20.csv']
for file in files:
  # diabetes dataset
    #data = np.loadtxt(file, delimiter=',')
    data = np.loadtxt(file, delimiter=',', skiprows= 1,  usecols=(0, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21) )
    [n,p] = np.shape(data)
    
    std_deviation = []
    mean = []
    #result = np.array(np.shape()
    
    
    sum_attributes = np.sum(data, axis=0)
    
    # Calculate the mean _for each feature
    for x in range(0,p) : 
        mean.append(sum_attributes[x] / n )
    
    # For _all the values of an attribute
    
    
    for i in range(0, p):
        attribute = data[:, i]
        std = 0
        #Calculate the standard deviation 
        for x in range(0,n) : 
            std = std + ( math.pow( (attribute[x]  - mean[i]), 2))
        
        std = math.sqrt( std / (n - 1) ) 
        
        for x in range(0, n): 
            data[x,i] = ( attribute[x] - mean[i]) / std
            
        
        #result =  np.stack((result , attribute), axis=-1)
        
#print(result)
    np.savetxt(file, data, delimiter=' ,', newline='\n')
        
        #Normalize _all values
        
        
        
        
    
    
