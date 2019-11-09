# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:58:51 2019

@author: Ranjan
"""

from subprocess import check_output
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings 
warnings.filterwarnings('ignore')
from math import ceil
#Plots
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score  
from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import parallel_coordinates
#Advanced optimization
from scipy import optimize as op

#Load Data
dataset = pd.read_excel("airdata.xlsx")
print(dataset.head())

#Data setup

Month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Sep', 'Oct', 'Nov', 'Dec']
#Number of examples
m = dataset.shape[0]
#Features
n = 7
#Number of classes
k = 12

X = np.ones((m,n + 1))
y = np.array((m,1))
X[:,1] = dataset['PM2.5'].values
X[:,2] = dataset['CO'].values
X[:,3] = dataset['NH3'].values
X[:,4] = dataset['NOx'].values

X[:,5] = dataset['NO'].values
X[:,6] = dataset['NO2'].values
X[:,7] = dataset['SO2'].values

#Labels
y = dataset['Month'].values

#Mean normalization
for j in range(n):
    X[:, j] = (X[:, j] - X[:,j].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
 
#Logistic Regression

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Regularized cost function
def regCostFunction(theta, X, y, _lambda = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    reg = (_lambda/(2 * m)) * np.sum(theta**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

#Regularized gradient function
def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    reg = _lambda * theta /m

    return ((1 / m) * X.T.dot(h - y)) + reg

#Optimal theta 
def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)
    
    return result.x

#Training

all_theta = np.zeros((k, n + 1))

#One vs all
i = 0
for f in Month:
    #set the labels in 0 and 1
    tmp_y = np.array(y_train == f, dtype = int)
    optTheta = logisticRegression(X_train, tmp_y, np.zeros((n + 1,1)))
    all_theta[i] = optTheta
    i += 1
    
#Predictions
P = sigmoid(X_test.dot(all_theta.T)) #probability for each flower
p = [Month[np.argmax(P[i, :])] for i in range(X_test.shape[0])]

print("Test Accuracy ", accuracy_score(y_test, p) * 100 , '%')

#Confusion Matrix
cfm = confusion_matrix(y_test, p, labels = Month)

sb.heatmap(cfm, annot = True, xticklabels = Month, yticklabels = Month);

