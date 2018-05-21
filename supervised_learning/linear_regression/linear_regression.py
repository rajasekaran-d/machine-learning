#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applying linear regression to multidimensional data 
Created on Mon Apr 23 12:15:57 2018

@author: rajasekaran.d
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# import dataset 
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# encoding categorical variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3],  sparse = False)
X = onehotencoder.fit_transform(X)

# Avoiding dummy variable trap
# It wont be a problem for scikit learn algorithm and gradient descent
# for the sake of uniformity we will use this for all methods 
X = X[:,1:]

# spiltting test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit the model using scikit 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting the output
y_predscikit = regressor.predict(X_test)

# calculating the r2 
d1 = y_test - y_predscikit 
d2 = y_test - y_test.mean()
r2 = 1 - (d1.dot(d1) / d2.dot(d2))
print("the r-squared for scikit learn algorithm is:", r2)

# fit the model using direct method
# adding a constant column ie x0 = 1
X_train = np.append(arr = np.ones(shape = (X_train.shape[0],1),dtype = np.int8), values = X_train, axis = 1)
X_test = np.append(arr = np.ones(shape = (X_test.shape[0],1),dtype = np.int8), values = X_test, axis = 1)

# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
theta = np.linalg.solve(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))

# predicting the output
y_preddirect = np.dot(X_test, theta)

# calculating the r2 
d1 = y_test - y_preddirect 
d2 = y_test - y_test.mean()
r2 = 1 - (d1.dot(d1) / d2.dot(d2))
print("the r-squared using direct method is:", r2)

# backward elimination
# It is also know as step wise regression 
# It is used to find statistically revelant features 
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Sl is the p value threshold and it is usually set as 5% 
SL = 0.05
X = np.append(arr = np.ones(shape = (X.shape[0],1),dtype = np.int8), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# Using these features models can be fitted using any of the above methods ... 





