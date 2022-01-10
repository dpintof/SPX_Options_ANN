#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:54:14 2021

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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from os import path
from contextlib import redirect_stdout


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..",
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
put_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(
    call_df.drop(["Option_Average_Price"], axis = 1), 
    call_df.Option_Average_Price, test_size = 0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(
    put_df.drop(["Option_Average_Price"], axis = 1), 
    put_df.Option_Average_Price, test_size = 0.01)


# Load models
call = tf.keras.models.load_model('Saved_models/mlp1_call_2')
put = tf.keras.models.load_model('Saved_models/mlp1_put_2') 


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


# Calculate error metrics
line1 = error_metrics(call_y_test, call.predict(call_X_test).reshape(
    call_y_test.shape[0]))
line2 = error_metrics(put_y_test, put.predict(put_X_test).reshape(
    put_y_test.shape[0]))

# Add train-MSE to the lists with the other risk metrics
line1.insert(0, np.mean(np.square(call_y_train - call.predict(
    call_X_train).reshape(call_y_train.shape[0]))))
line2.insert(0, np.mean(np.square(put_y_train - put.predict(
    put_X_train).reshape(put_y_train.shape[0]))))

metric_names = ["Train MSE", "MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", 
                "PE20"]

metric_dictionary1 = {metric_names[i]: line1[i] for i in range(len(
    metric_names))}
metric_dictionary2 = {metric_names[i]: line2[i] for i in range(len(
    metric_names))}

print("Error metrics for call options")
for key, value in metric_dictionary1.items():
    print(f"{key}:", value)

with open('MLP1_call_error_metrics.txt', 'w') as f:
    with redirect_stdout(f):
        print("Error metrics for call options")
        for key, value in metric_dictionary1.items():
            print(f"{key}:", value)

print("\nError metrics for put options")
for key, value in metric_dictionary2.items():
 	    print(f"{key}:", value)

with open('MLP1_put_error_metrics.txt', 'w') as f:
    with redirect_stdout(f):
        print("Error metrics for put options")
        for key, value in metric_dictionary2.items():
            print(f"{key}:", value)

