#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:27:58 2021

@author: Diogo
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from scipy.stats import norm
from os import path


N_TIMESTEPS = 20


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options-df.csv"))
options_df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
options_df = options_df.drop(columns=['Sigma_20_Days_Annualized', 
                                      "Underlying_Price", "bid_eod", "ask_eod"])
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/underlying_df.csv"))
underlying = pd.read_csv(filepath)

# Create array with the prices of the underlying, where the first N_TIMESTEPS 
    # entries are nan
padded = np.insert(underlying[" Close"].values, 0, 
                   np.array([np.nan] * N_TIMESTEPS))

# Create Dataframe (df) where the first column is a date and the rest are 
    # prices of the underlying. Each row has N_TIMESTEPS prices of the 
    # underlying, ordered by date, in descending order, from left to right. 
    # From one row to the next each price is replaced by the one observed in 
    # the next date.
rolled = np.column_stack([np.roll(padded, i) for i in range(N_TIMESTEPS)])
rolled = rolled[~np.isnan(rolled).any(axis=1)]
rolled = np.column_stack((underlying.Date.values[N_TIMESTEPS - 1:], rolled))
price_history = pd.DataFrame(data=rolled)

cols = price_history.columns.drop(0)
price_history[cols] = price_history[cols].apply(pd.to_numeric, errors='coerce')
    # The last 2 rows are necessary because if the prices of the underlying are
        # not in a numeric data type, Keras' fit function will not work.

# Add columns of price_history to options_df, according to the date in "QuoteDate"
joined = options_df.join(price_history.set_index(0), on='QuoteDate')

# Create dfs for calls and puts
call_df = joined[joined.OptionType == 'c'].drop(['OptionType'], axis=1)
call_df = call_df.drop(columns=['QuoteDate'])
put_df = joined[joined.OptionType == 'p'].drop(['OptionType'], axis=1)
put_df = put_df.drop(columns=['QuoteDate'])

# Split dfs into random train and test arrays, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Option_Average_Price"], 
        axis=1).values, call_df.Option_Average_Price.values, test_size=0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(["Option_Average_Price"], 
        axis=1).values, put_df.Option_Average_Price.values, test_size=0.01)

# Create lists composed of a 3-dimensional array containing the N_TIMESTEPS 
    # prices of the underlying per row and an array with n_feature input values
    # per row.
call_X_train = [call_X_train[:, -N_TIMESTEPS:].reshape(call_X_train.shape[0], 
                                        N_TIMESTEPS, 1), call_X_train[:, :4]]
    # call_X_train[:, -N_TIMESTEPS:] returns an array with the last N_TIMESTEPS
        # columns of call_X_train. Meaning the columns with the prices of the 
        # underlying.
    # call_X_train.shape[0] returns the number of rows of call_X_train
    # call_X_train[:, :n_features] returns an array with the n_features 
        # columns of call_X_train.
call_X_test = [call_X_test[:, -N_TIMESTEPS:].reshape(call_X_test.shape[0], 
                                        N_TIMESTEPS, 1), call_X_test[:, :4]]
put_X_train = [put_X_train[:, -N_TIMESTEPS:].reshape(put_X_train.shape[0], 
                                        N_TIMESTEPS, 1), put_X_train[:, :4]]
put_X_test = [put_X_test[:, -N_TIMESTEPS:].reshape(put_X_test.shape[0], 
                                        N_TIMESTEPS, 1), put_X_test[:, :4]]


call_model = tf.keras.models.load_model('Saved_models/lstm_call_1')
put_model = tf.keras.models.load_model('Saved_models/lstm_put_1')


# def black_scholes_call(row):
#     S = row.Underlying_Price
#     X = row.Strike
#     T = row.Time_to_Maturity
#     r = row.RF_Rate / 100
#     σ = row.Historical_Vol
#     d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
#     d2 = d1 - σ * (T ** .5)
#     c = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
#     return c

# def black_scholes_put(row):
#     S = row.Underlying_Price
#     X = row.Strike
#     T = row.Time_to_Maturity
#     r = row.RF_Rate / 100
#     σ = row.Historical_Vol
#     d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
#     d2 = d1 - σ * (T ** .5)
#     p  = norm.cdf(-d2) * X * np.exp(-r * T) - S * norm.cdf(-d1)
#     return p

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

line1 = error_metrics(call_y_test, call_model.predict(call_X_test, 
                                batch_size=4096).reshape(call_y_test.shape[0]))
line2 = error_metrics(put_y_test, put_model.predict(put_X_test, 
                                batch_size=4096).reshape(put_y_test.shape[0]))

line1.insert(0, np.mean(np.square(call_y_train - call_model.predict(call_X_train).reshape(call_y_train.shape[0]))))
line2.insert(0, np.mean(np.square(put_y_train - put_model.predict(put_X_train).reshape(put_y_train.shape[0]))))

metric_names = ["Train MSE", "MSE", "Bias", "AAPE", "MAPE", "PE5", "PE10", 
                "PE20"]

metric_dictionary1 = {metric_names[i]: line1[i] for i in range(len(metric_names))}
metric_dictionary2 = {metric_names[i]: line2[i] for i in range(len(metric_names))}

print("Error metrics for call options, regarding the ANN's test sample and "
      "respective predictions")
for key, value in metric_dictionary1.items():
	    print(f"{key}:", value)

print("\nError metrics for put options, regarding the ANN's test sample and "
      "respective predictions")
for key, value in metric_dictionary2.items():
 	    print(f"{key}:", value)

