#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:55:49 2021

@author: Diogo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import kde
from keras.models import load_model
from os import path


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "options-df.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
put_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["bid_eod",
        "ask_eod"], axis = 1), call_df[["bid_eod", "ask_eod"]], test_size = 0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(["bid_eod",
        "ask_eod"], axis = 1), put_df[["bid_eod", "ask_eod"]], test_size = 0.01)


call = load_model('Saved_models/mlp2_call_3')
# call = load_model('mlp2_call_5') # TESTING WITH A SMALL SAMPLE
put = load_model('Saved_models/mlp2_put_3')


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

call_y_pred = call.predict(call_X_test)
put_y_pred = put.predict(put_X_test)

call_errors = error_metrics(np.mean(call_y_test, axis=1), np.mean(call_y_pred, 
                                                                  axis=1))
# np.mean is used because we want to calculate errors on the average between 
    # bid and ask prices.
put_errors = error_metrics(np.mean(put_y_test, axis=1), np.mean(put_y_pred, 
                                                                axis=1))


def matrix(actual, predicted, q):
    rel = (actual - predicted) / actual
    def segregate(x, q): # results in either -2 or -1
        up = x > q 
        low = x < -q
        mid = ~(up | low) # ~ and | operators: 
            # https://wiki.python.org/moin/BitwiseOperators
        return (up, mid, low)
    bid = rel.iloc[:,0] # series with all the relative bid prices
    ask = rel.iloc[:,1] # series with all the relative ask prices
    x = segregate(bid, q)
    y = segregate(ask, q)
    return np.array([[sum(x[i] & y[j]) for i in range(3)] for j in range(3)]) / rel.shape[0]
    # rel.shape[0] returns the number of rows in the rel df

matrix(call_y_test, call_y_pred, 0.01)
matrix(call_y_test, call_y_pred, 0.05)
matrix(put_y_test, put_y_pred, 0.01) 
matrix(put_y_test, put_y_pred, 0.05)


def error_scatter(actual, predicted):
    temp = 100 * (actual - predicted) / actual
    plt.scatter(temp.iloc[:,0], temp.iloc[:,1], s=1)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.xlabel('Bid Error %')
    plt.ylabel('Ask Error %')

error_scatter(call_y_test, call_y_pred)
plt.title('MLP2 Call Percent Errors')
plt.savefig('Saved_graphs/mlp2_call_scatter.png')

error_scatter(put_y_test, put_y_pred)
plt.title('MLP2 Put Percent Errors')
plt.savefig('Saved_graphs/mlp2_put_scatter.png')


def kde_scatter(actual, predicted):
    rel = 100 * (actual - predicted) / actual
    rel = rel.replace([np.inf, -np.inf], np.nan)
    rel = rel.dropna()
    temp = rel[np.linalg.norm(rel, ord=np.inf, axis=1) < 100]
    k = kde.gaussian_kde(temp.T)
    plt.scatter(temp.iloc[:,0], temp.iloc[:,1], c=k(temp.T), s=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('Bid Error %')
    plt.ylabel('Ask Error %')

# kde_scatter(call_y_test, call_y_pred) # PROVAVELMENTE ESTÁ A DAR ERRO POIS ESTOU A USAR UM MODELO DE TESTES
# plt.title('MLP2 Call Percent Errors')
# plt.savefig('Saved_graphs/mlp2_call_kde.png')

# kde_scatter(put_y_test, put_y_pred)
# plt.title('MLP2 Put Percent Errors')
# plt.savefig('Saved_graphs/mlp2_put_kde.png')


call_train_pred = call.predict(call_X_train)
put_train_pred = put.predict(put_X_train)

# Add train-MSE to the lists with the other risk metrics
call_errors.insert(0, np.mean(np.square(np.diff(call_y_train, 
                                axis=1) - np.diff(call_train_pred, axis=1))))
put_errors.insert(0, np.mean(np.square(np.diff(put_y_train, 
                                axis=1) - np.diff(put_train_pred, axis=1))))

metric_names = ["Train MSE", "MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", 
                "PE20"]

metric_dictionary1 = {metric_names[i]: call_errors[i] for i in range(len(metric_names))}
metric_dictionary2 = {metric_names[i]: put_errors[i] for i in range(len(metric_names))}

print("Error metrics for call options, regarding the ANN's test sample and "
      "respective predictions")
for key, value in metric_dictionary1.items():
	    print(f"{key}:", value)
        
print("\nError metrics for put options, regarding the ANN's test sample and "
      "respective predictions")
for key, value in metric_dictionary2.items():
 	    print(f"{key}:", value)

