#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:33:44 2021

@author: Diogo
"""

"""Clear the console and remove all variables present on the namespace. This is 
useful to prevent Python from consuming more RAM each time I run the code."""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pandas as pd
import numpy as np
import math


call = pd.read_csv("BSM_predictions/bsm_call.csv")
put = pd.read_csv("BSM_predictions/bsm_put.csv")


def error_metrics(actual, predicted):
    diff = actual - predicted
    mse = np.mean(np.square(diff))
    rel = diff / actual
    bias = 100 * np.median(rel)
    # bias = 100 * median(rel)
    aape = 100 * np.mean(np.abs(rel))
    # aape = 100 * (np.abs(rel)).sum() / rel.shape[0]
    mape = 100 * np.median(np.abs(rel))
    # mape = 100 * median(np.abs(rel))
    pe5 = 100 * sum(np.abs(rel) < 0.05) / rel.shape[0]
    pe10 = 100 * sum(np.abs(rel) < 0.10) / rel.shape[0]
    pe20 = 100 * sum(np.abs(rel) < 0.20) / rel.shape[0]
    return [mse, bias, aape, mape, pe5, pe10, pe20]

line1 = error_metrics(call.Option_Average_Price, call.BSM_Prediction)
line2 = error_metrics(put.Option_Average_Price, put.BSM_Prediction)

metric_names = ["MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", "PE20"]

call_metric_dictionary = {metric_names[i]: line1[i] for i in range(len(metric_names))}
put_metric_dictionary = {metric_names[i]: line2[i] for i in range(len(metric_names))}

print("Error metrics for call options, regarding the average price between ask"
      " and bid, and the BSM model's prediction")
for key, value in call_metric_dictionary.items():
	    print(f"{key}:", value)

print("\nError metrics for put options, regarding the average price between "
      "ask and bid, and the BSM model's prediction")
for key, value in put_metric_dictionary.items():
 	    print(f"{key}:", value)

"""Save metrics into files"""
with open("bsm_call_metrics.txt", "w") as f:
    for key, value in call_metric_dictionary.items():
        f.write('%s:%s\n' % (key, value))
        
with open("bsm_put_metrics.txt", "w") as f:
    for key, value in put_metric_dictionary.items():
        f.write('%s:%s\n' % (key, value))

