#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:34:27 2021

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


import numpy as np
import pandas as pd
from scipy.stats import norm
from os import path
from tqdm import tqdm


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
options_df = pd.read_csv(filepath)
call_df = options_df[options_df.OptionType == 'c'].drop(['OptionType'], axis=1)
put_df = options_df[options_df.OptionType == 'p'].drop(['OptionType'], axis=1)


def black_scholes_call(row):
    S = row.Underlying_Price
    X = row.strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Sigma_20_Days_Annualized
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    c = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return c

def black_scholes_put(row):
    S = row.Underlying_Price
    X = row.strike
    T = row.Time_to_Maturity
    r = row.RF_Rate / 100
    σ = row.Sigma_20_Days_Annualized
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    p  = norm.cdf(-d2) * X * np.exp(-r * T) - S * norm.cdf(-d1)
    return p


# Total number of call and put options, respectively.
n_call = call_df.shape[0]
n_put = put_df.shape[0]

# """Add BSM_Predictio column to call_df and put_df"""
# bsm_prediction_list = []
# for index, row in tqdm(call_df.iterrows(), total = n_call):
#     bsm_prediction_list.append(black_scholes_call(row))

# call_df['BSM_Prediction'] = bsm_prediction_list

"""Add BSM_Prediction column to both call_df and put_df"""
tqdm.pandas()
print("2 lengthy commands will follow, with respective progress bars")
call_df['BSM_Prediction'] = call_df.progress_apply(black_scholes_call, 
                                                   axis = 1)
put_df['BSM_Prediction'] = put_df.progress_apply(black_scholes_put, axis = 1)
# call_df['BSM_Prediction'] = call_df.apply(black_scholes_call, axis=1)
# put_df['BSM_Prediction'] = put_df.apply(black_scholes_put, axis=1)

call_df = call_df.drop(columns=["QuoteDate", "strike", "bid_eod", "ask_eod", 
                          "Time_to_Maturity", "RF_Rate", 
                          "Sigma_20_Days_Annualized", "Underlying_Price"])
put_df = put_df.drop(columns=["QuoteDate", "strike", "bid_eod", "ask_eod", 
                          "Time_to_Maturity", "RF_Rate", 
                          "Sigma_20_Days_Annualized", "Underlying_Price"])

call_df.to_csv('BSM_predictions/bsm_call.csv', index=False)
put_df.to_csv('BSM_predictions/bsm_put.csv', index=False)

