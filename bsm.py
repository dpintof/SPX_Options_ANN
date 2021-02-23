#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:34:27 2021

@author: Diogo
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
# from sklearn.model_selection import train_test_split


# df = pd.read_csv('daily-closing-prices.csv')
# estimate_σ = lambda arr: (np.diff(arr) / arr[:-1]).std()
# df['sigma_20'] = df.close.rolling(20).apply(estimate_σ)
# date_sigma = df.drop(['close'], axis=1)
df = pd.read_csv("options-df.csv")
# options_df = pd.read_csv('options-df.csv').drop(['impl_volatility', 'exdate'], axis=1)
# options_df_with_sigma = options_df.set_index('date').join(date_sigma.set_index('date'))
# options_df_new = options_df_with_sigma.dropna(axis=0)
call = df[df.OptionType == 'call'].drop(['OptionType'], axis=1)
put = df[df.OptionType == 'put'].drop(['OptionType'], axis=1)


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


call['BSM_Prediction'] = call.apply(black_scholes_call, axis=1)
put['BSM_Prediction'] = put.apply(black_scholes_put, axis=1)

call = call.drop(columns=["QuoteDate", "Strike", "Bid", "Ask", 
                          "Time_to_Maturity", "RF_Rate", "Historical_Vol", 
                          "Underlying_Price"])
put = put.drop(columns=["QuoteDate", "Strike", "Bid", "Ask", 
                          "Time_to_Maturity", "RF_Rate", "Historical_Vol", 
                          "Underlying_Price"])

call.to_csv('bsm_call.csv', index=False)
put.to_csv('bsm_put.csv', index=False)

