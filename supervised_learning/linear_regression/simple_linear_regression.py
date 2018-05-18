#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applying linear regression to one dimensional data 

Created on Thu Apr 19 11:43:27 2018
@author: rajasekaran.d
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# load data 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# splitting the dataset into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# fitting training data using algorithm from scikit learn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting output 
y_predscikit = regressor.predict(X_test)

# plotting the result for both training and test set
plt.scatter(X_train,y_train, color = "Red")
plt.plot(X_train,regressor.predict(X_train),color = "Blue")
plt.title("Simple Regression using scikit learn (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test, color = "Red")
plt.plot(X_train,regressor.predict(X_train),color = "Blue")
plt.title("Simple Regression using scikit learn (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# computing the r2 
d1scikit = y_test - y_predscikit 
d2scikit = y_test - y_test.mean()
r2scikit = 1 - (d1scikit.dot(d1scikit) / d2scikit.dot(d2scikit))
print("the r-squared using scikit is:", r2scikit)

# solving linear regression by ourselves without using any external api
# calculating a and b to find the line of best fit
# line of best fit y = ax + b
# a and b found by equating the derivative of cost function to zero
# cost function is sum of square of residuals
# making X_train and X_test as 1 D array
X_train = X_train[:,0]
X_test = X_test[:,0]
xy = np.dot(X_train,y_train)
xmean = X_train.mean()
ymean = y_train.mean()
xsum = X_train.sum()
ysum = y_train.sum()
xdot = np.dot(X_train,X_train)

denom = xdot - (xmean * xsum)
a = (xy - (ymean * xsum)) / denom
b =  ((ymean * xdot) - (xmean * xy)) / denom

#predicting the output using the line of fit
y_pred = a * X_test + b

# plotting the result for both training and test set
plt.scatter(X_train,y_train, color = "Red")
plt.plot(X_train,a * X_train + b,color = "Blue")
plt.title("Simple Regression (Training Set)") 
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test, color = "Red")
plt.plot(X_train,a * X_train + b,color = "Blue")
plt.title("Simple Regression (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# calculating the r2
d1 = y_test - y_pred 
d2 = y_test - y_test.mean()
r2 = 1 - (d1.dot(d1) / d2.dot(d2))
print("the r-squared is:", r2)
