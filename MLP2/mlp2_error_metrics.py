#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:55:49 2021

@author: Diogo
"""

# Clear the console and remove all variables present on the namespace
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
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
put_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)

# Remove rows with at least one 0 to prevent division by 0.
call_df = call_df[call_df["bid_eod"] != 0]
call_df = call_df[call_df["ask_eod"] != 0]


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(
    call_df.drop(["bid_eod", "ask_eod"], axis = 1), 
    call_df[["bid_eod", "ask_eod"]], test_size = 0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(
    put_df.drop(["bid_eod", "ask_eod"], axis = 1), 
    put_df[["bid_eod", "ask_eod"]], test_size = 0.01)

call_y_train = call_y_train.mean(axis=1).reset_index(drop=True)
call_y_test = call_y_test.mean(axis=1).reset_index(drop=True)
put_y_train = put_y_train.mean(axis=1).reset_index(drop=True)
put_y_test = put_y_test.mean(axis=1).reset_index(drop=True)


# Load models
call = tf.keras.models.load_model('Saved_models/mlp2_call_1')
put = tf.keras.models.load_model('Saved_models/mlp2_put_1') 


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
call_y_pred = pd.DataFrame(call.predict(call_X_test)).mean(axis=1)
put_y_pred = pd.DataFrame(put.predict(put_X_test)).mean(axis=1)

call_error_metrics = error_metrics(call_y_test, call_y_pred)
put_error_metrics = error_metrics(put_y_test, put_y_pred)


# Add train-MSE to the list with the other risk metrics
call_y_pred_train = pd.DataFrame(call.predict(call_X_train)).mean(axis=1)
call_y_train.reset_index(drop=True, inplace=True)
put_y_pred_train = pd.DataFrame(put.predict(put_X_train)).mean(axis=1)
put_y_train.reset_index(drop=True, inplace=True)

call_error_metrics.insert(0, np.mean(np.square(call_y_train - call_y_pred_train)))
put_error_metrics.insert(0, np.mean(np.square(put_y_train - put_y_pred_train)))


# Print error metrics and save to file
metric_names = ["Train MSE", "MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", 
                "PE20"]

call_metric_dict = {metric_names[i]: call_error_metrics[i] for i in range(len(
                                                                metric_names))}
put_metric_dict = {metric_names[i]: put_error_metrics[i] for i in range(len(
                                                                metric_names))}

print("Error metrics for call options")
for key, value in call_metric_dict.items():
    print(f"{key}:", value)

with open('MLP2_call_error_metrics.txt', 'w') as f:
    with redirect_stdout(f):
        print("Error metrics for call options")
        for key, value in call_metric_dict.items():
            print(f"{key}:", value)

print("\nError metrics for put options")
for key, value in put_metric_dict.items():
 	    print(f"{key}:", value)

with open('MLP2_put_error_metrics.txt', 'w') as f:
    with redirect_stdout(f):
        print("Error metrics for put options")
        for key, value in put_metric_dict.items():
            print(f"{key}:", value)

