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
from os import path


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Processed data/options-df.csv"))
options_df = pd.read_csv(filepath)
call_df = options_df[options_df.OptionType == 'c'].drop(['OptionType'], axis=1)
put_df = options_df[options_df.OptionType == 'p'].drop(['OptionType'], axis=1)


def black_scholes_call(row):
    S = row.Underlying_Price
    X = row.strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Sigma_20_Days
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    c = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return c

def black_scholes_put(row):
    S = row.Underlying_Price
    X = row.strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Sigma_20_Days
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    p  = norm.cdf(-d2) * X * np.exp(-r * T) - S * norm.cdf(-d1)
    return p


call_df['BSM_Prediction'] = call_df.apply(black_scholes_call, axis=1)
put_df['BSM_Prediction'] = put_df.apply(black_scholes_put, axis=1)

call_df = call_df.drop(columns=["QuoteDate", "strike", "bid_eod", "ask_eod", 
                          "Time_to_Maturity", "RF_Rate", "Sigma_20_Days", 
                          "Underlying_Price"])
put_df = put_df.drop(columns=["QuoteDate", "strike", "bid_eod", "ask_eod", 
                          "Time_to_Maturity", "RF_Rate", "Sigma_20_Days", 
                          "Underlying_Price"])

call_df.to_csv('BSM predictions/bsm_call.csv', index=False)
put_df.to_csv('BSM predictions/bsm_put.csv', index=False)

