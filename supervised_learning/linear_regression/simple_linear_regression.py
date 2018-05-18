#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:43:27 2018

@author: raja-4457
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import math

# import dataset 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

"""# fitting training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting output
y_pred = regressor.predict(X_test)

# plotting the result

plt.scatter(X_test,y_test, color = "Red")
plt.plot(X_train,regressor.predict(X_train),color = "Blue")
plt.title("Simple Regression")
plt.xlabel("Years of experience")
plt.ylabel("Salary")

plt.scatter(X_train,y_train, color = "Red")
plt.plot(X_train,ypred,color = "Blue")"""
xy = np.dot(X_train,y_train)
xmean = X_train.mean()
ymean = y_train.mean()
xsum = X_train.sum()
ysum = y_train.sum()
xdot = np.dot(X_train,X_train)

denom = np.dot(X_train,X_train) - math.pow((X_train.mean()), 2)
a = np.dot(X_train,y_train) - X_train.mean() * y_train.mean() / denom
b = y_train.mean() * math.pow((X_train.mean()), 2) - X_train.mean() * np.dot(X_train,y_train) / denom

ypred = a * X_train + b

denom1 = xdot - (xmean * xsum)
a1 = (xy - (ymean * xsum)) / denom1
b1 = ((ymean * xdot) - (xmean * xy)) / denom1
