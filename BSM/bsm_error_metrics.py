#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:33:44 2021

@author: Diogo
"""

import pandas as pd
import numpy as np


call = pd.read_csv("BSM predictions/bsm_call.csv")
put = pd.read_csv("BSM predictions/bsm_put.csv")
 
# Split data into train and test sets
# call_train = call.sample(frac=0.99)
# call_test = call.loc[~call.index.isin(call_train.index)]
# put_train = put.sample(frac=0.99)
# put_test = put.loc[~put.index.isin(put_train.index)]


def error_metrics(actual, predicted):
    diff = actual - predicted
    mse = np.mean(np.square(diff))
    rel = diff / actual
    bias = 100 * np.median(rel)
    aape = 100 * np.mean(np.abs(rel))
    mape = 100 * np.median(np.abs(rel))
    pe5 = 100 * sum(np.abs(rel) < 0.05) / rel.shape[0]
    pe10 = 100 * sum(np.abs(rel) < 0.10) / rel.shape[0]
    pe20 = 100 * sum(np.abs(rel) < 0.20) / rel.shape[0]
    return [mse, bias, aape, mape, pe5, pe10, pe20]

line1 = error_metrics(call.Option_Average_Price, call.BSM_Prediction)
line2 = error_metrics(put.Option_Average_Price, put.BSM_Prediction)

# line1.insert(0, np.mean(np.square(call_y_train - call_model.predict(call_X_train).reshape(call_y_train.shape[0]))))
# line2.insert(0, np.mean(np.square(put_y_train - put_model.predict(put_X_train).reshape(put_y_train.shape[0]))))

metric_names = ["MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", "PE20"]

metric_dictionary1 = {metric_names[i]: line1[i] for i in range(len(metric_names))}
metric_dictionary2 = {metric_names[i]: line2[i] for i in range(len(metric_names))}

print("Error metrics for call options, regarding the average price between ask"
      " and bid, and the BSM model's prediction")
for key, value in metric_dictionary1.items():
	    print(f"{key}:", value)

print("\nError metrics for put options, regarding the average price between "
      "ask and bid, and the BSM model's prediction")
for key, value in metric_dictionary2.items():
 	    print(f"{key}:", value)

