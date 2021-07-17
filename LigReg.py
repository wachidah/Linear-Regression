#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


#Creating a class of Linear Regression

class LinReg:
  def __init__(self, x, y):
    self.input = x
    self.label = y
    self.weight = 0
    self.bias = 0
    self.n = len(x)
  def fit(self , epoch , learning_rate):
    for i in range(epoch):
      y_predict = self.weight * self.input + self.bias
      #Calculating derivatives
      D_weight = (-2/self.n)*sum(self.input * (self.label - y_predict))
      D_bias = (-1/self.n)*sum(self.label-y_predict)
      #Updating Parameters
      self.weight = self.weight - learning_rate * D_weight
      self.c = self.bias - learning_rate * D_bias
  def predict(self , inputs):
      y_predict = self.weight * inputs + self.bias 
      return y_predict

# Create linear regression object
regr = LinReg(diabetes_X_train, diabetes_y_train)

# Train the model using number of epochs and learning rate
regr.fit(10000, 0.0004)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
x_mean = np.mean(diabetes_X_test)
y_mean = np.mean(diabetes_y_pred)
#total number of values
num = len(diabetes_X_test)
#Calculating b1 and b0
numerator = 0
denominator = 0
for i in range(num):
    numerator += (diabetes_X_test[i] - x_mean) * (diabetes_y_pred[i] - y_mean)
    denominator += (diabetes_X_test[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

#Calculating MSE (Mean Square Error)
Loss = np.sum(pow((diabetes_y_pred-diabetes_X_test), 2))
MSE = Loss/num

#Calculating R2 Score
SSt = np.sum((diabetes_X_test - x_mean)**2)
R2 = 1-(SSt/Loss)

#printing the coefficient
print('Coefficients: B0=, B1=' , b1, b0)
print('Intercept: ', b0)
# The mean squared error
print('Mean squared error:', MSE)
# The coefficient of determination (R2 Score)
print('Coefficient of determination (R2 Score):', R2)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
plt.plot(diabetes_X_test, diabetes_y_pred, color='black')
plt.show()






