# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:35:55 2025

@author: Sunil Kumar SJ
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
def rmse_metrics(actual,predicted):
    predicted_error=0
    n=len(actual)
    for y_act,y_predicted in zip(actual,predicted):
        predicted_error+=(y_act-y_predicted)**2
    mean_error=predicted_error/float(n)
    return sqrt(mean_error)

def compute_coefficient(x,y):
    #Here we have to get slope and intercept of the line
    # slope is denoted by w1
    #intercept or constant is denoted by w0
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    numerator=0;
    denominator=0;
    n=len(x);
    for i in range(n):
        numerator+=(x[i]-x_mean)*(y[i]-y_mean)
        denominator+=(x[i]-x_mean)**2
    slope=numerator/denominator
    intercept=y_mean-(slope*x_mean)
    return slope,intercept

def predicted(x,w0,w1):
    return x*w1+w0 #y=x1*w1+x2*w2+x3*w3+......+xn*wn+w0

def evaluate_OLS(y,y_hat):
    mse=np.mean((y-y_hat)**2)
    return mse,sqrt(mse)
x = np.arange(1, 51)
y = x*3+5

# Add some random error to the array
y[np.random.randint(0, len(y), size=10)] += np.random.randint(-5, 5)

w1, w0 = compute_coefficient(x, y)
y_hat = predicted(x,w1,w0)
data=evaluate_OLS(y, y_hat)
print("data",data)
# display the value of predicted coefficients
print(w1,w0)

plt.scatter(x, y, label='Observed Value')
plt.plot(x, y_hat, label='Predicted Value', color='red')
plt.xlabel('<--X-Axis-->')
plt.ylabel('<--Y-Axis-->')
plt.legend()
plt.show()


