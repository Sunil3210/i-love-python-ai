# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:35:55 2025

@author: Sunil Kumar SJ
"""
from math import sqrt
def rmse_metrics(actual,predicted):
    predicted_error=0
    n=len(actual)
    for y_act,y_predicted in zip(actual,predicted):
        predicted_error+=(y_act-y_predicted)**2
    mean_error=predicted_error/float(n)
    return sqrt(mean_error)

