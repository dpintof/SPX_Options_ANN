#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:54:14 2021

@author: Diogo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from scipy.stats import norm


df = pd.read_csv("options-df.csv")
# df = df.dropna(axis=0)
# df.strike_price = df.strike_price / 1000
call_df = df[df.OptionType == 'call'].drop(['OptionType'], axis=1)
put_df = df[df.OptionType == 'put'].drop(['OptionType'], axis=1)


call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Average_Price"],
                            axis = 1), call_df.Average_Price, test_size = 0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(call_df.drop(["Average_Price"],
                            axis = 1), call_df.Average_Price, test_size = 0.01)


call = load_model('mlp1_call_3')
# put = load_model('mlp1_put_3')


def black_scholes_call(row):
    S = row.Underlying_Price
    X = row.Strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Historical_Vol
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    c = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return c

def black_scholes_put(row):
    S = row.Underlying_Price
    X = row.Strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Historical_Vol
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    p  = norm.cdf(-d2) * X * np.exp(-r * T) - S * norm.cdf(-d1)
    return p


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


line1 = error_metrics(call_y_test, call.predict(call_X_test).reshape(call_y_test.shape[0]))
# line2 = error_metrics(put_y_test, put.predict(put_X_test).reshape(put_y_test.shape[0]))
line3 = error_metrics(call_y_test, black_scholes_call(call_X_test))
# line4 = error_metrics(put_y_test, black_scholes_put(put_X_test))

line3.insert(0, np.mean(np.square(call_y_train - black_scholes_call(call_X_train))))
# line4.insert(0, np.mean(np.square(put_y_train - black_scholes_put(put_X_train))))
line1.insert(0, np.mean(np.square(call_y_train - call.predict(call_X_train).reshape(call_y_train.shape[0]))))
# line2.insert(0, np.mean(np.square(put_y_train - put.predict(put_X_train).reshape(put_y_train.shape[0]))))


# for line in (line1, line2, line3, line4):
#     print('& {:.2f} & {:.2f} & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% \\\\'.format(*line))
for line in (line1, line3): # TESTING ONLY FOR CALL MODEL. NEED TO FORMAT THE OUTPUT.
    print('& {:.2f} & {:.2f} & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% \\\\'.format(*line))  
for line in (line1, line3):
    print(line)
    