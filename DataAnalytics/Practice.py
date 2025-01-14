# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:42:33 2025

@author: ssj55
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
def rmse_metrics(actual,predicted):
    sum_squared_error=0.0;
    n=len(actual)
    for i in range(n):
        sum_squared_error+=np.sum((actual[i]-predicted[i])**2)
    mean_error=sum_squared_error/n
    return mean_error



def compute_coefficient(x,y):
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    numerator=0.0
    denominator=0.0
    n=len(y)
    for i in range(n):
        numerator+=(x[i]-x_mean)*(y[i]-y_mean)
        denominator+=(x[i]-x_mean)**2
    slope=numerator/denominator
    intercept=y_mean-(slope)*x_mean
    return slope,intercept

def predict(x,m,c):
    return m*x+c
x = np.arange(1, 51)
y_actual=2*x+3
y_actual[np.random.randint(0, len(y_actual), size=10)] += np.random.randint(-5, 5)
print("x is",x)
print("y is with some error",y_actual)

slope,intercept=compute_coefficient(x, y_actual)
y_predicted=predict(x, slope, intercept)
print("predicted y ",y_predicted)

plt.scatter(x, y_actual,label='observed Value')
plt.plot(x, y_predicted,label='predicted Value')
plt.xlabel("<-----x-axis----->")
plt.ylabel("<-----y-axis----->")


