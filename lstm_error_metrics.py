#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:27:58 2021

@author: Diogo
"""

# from keras.models import Sequential, Model, load_model
from keras.models import load_model
# from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, LSTM, Bidirectional, Input, Concatenate
# from keras import backend as K
# from keras.callbacks import TensorBoard
# from keras.optimizers import Adam
# from keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('../options-df-sigma.csv')
df = df.dropna(axis=0)
df = df.drop(columns=['exdate', 'impl_volatility', 'volume', 'open_interest', 'sigma_20'])
df.strike_price = df.strike_price / 1000
call_df = df[df.cp_flag == 'C'].drop(['cp_flag'], axis=1)
put_df = df[df.cp_flag == 'P'].drop(['cp_flag'], axis=1)
underlying = pd.read_csv('../daily-closing-prices.csv')
N_TIMESTEPS = 20
padded = np.insert(underlying.close.values, 0, np.array([np.nan] * N_TIMESTEPS))
rolled = np.column_stack([np.roll(padded, i) for i in range(N_TIMESTEPS)])
rolled = rolled[~np.isnan(rolled).any(axis=1)]
rolled = np.column_stack((underlying.date.values[N_TIMESTEPS - 1:], rolled))
price_history = pd.DataFrame(data=rolled)
joined = df.join(price_history.set_index(0), on='date')
call_df = joined[joined.cp_flag == 'C'].drop(['cp_flag'], axis=1)
put_df = joined[joined.cp_flag == 'P'].drop(['cp_flag'], axis=1)
call_df = call_df.drop(columns=['date'])
put_df = put_df.drop(columns=['date'])
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(['best_bid', 'best_offer'], axis=1).values,
                                                                        ((call_df.best_bid + call_df.best_offer) / 2).values,
                                                                        test_size=0.01, random_state=42)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(['best_bid', 'best_offer'], axis=1).values,
                                                                    ((put_df.best_bid + put_df.best_offer) / 2).values,
                                                                    test_size=0.01, random_state=42)
call_X_train = [call_X_train[:, -N_TIMESTEPS:].reshape(call_X_train.shape[0], N_TIMESTEPS, 1), call_X_train[:, :4]]
call_X_test = [call_X_test[:, -N_TIMESTEPS:].reshape(call_X_test.shape[0], N_TIMESTEPS, 1), call_X_test[:, :4]]
put_X_train = [put_X_train[:, -N_TIMESTEPS:].reshape(put_X_train.shape[0], N_TIMESTEPS, 1), put_X_train[:, :4]]
put_X_test = [put_X_test[:, -N_TIMESTEPS:].reshape(put_X_test.shape[0], N_TIMESTEPS, 1), put_X_test[:, :4]]


# In[3]:


call_model = load_model('saved-models/20191207-call-lstm-v3.h5')
put_model = load_model('saved-models/20191207-put-lstm-v3.h5')


# In[8]:


from scipy.stats import norm
def black_scholes(row):
    S = row.closing_price
    X = row.strike_price
    T = row.date_ndiff / 365
    r = row.treasury_rate / 100
    σ = row.sigma_20
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    C = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return C
def black_scholes_put(row):
    S = row.closing_price
    X = row.strike_price
    T = row.date_ndiff / 365
    r = row.treasury_rate / 100
    σ = row.sigma_20
    d1 = (np.log(S / X) + (r + (σ ** 2) / 2) * T) / (σ * (T ** .5))
    d2 = d1 - σ * (T ** .5)
    P  = norm.cdf(-d2) * X * np.exp(-r * T) - S * norm.cdf(-d1)
    return P


# In[4]:


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


# In[5]:


line1 = error_metrics(call_y_test, call_model.predict(call_X_test, batch_size=4096).reshape(call_y_test.shape[0]))
line2 = error_metrics(put_y_test, put_model.predict(put_X_test, batch_size=4096).reshape(put_y_test.shape[0]))


# In[16]:


call_model.evaluate(call_X_train, call_y_train, batch_size=4096)


# In[17]:


put_model.evaluate(put_X_train, put_y_train, batch_size=4096)


# In[18]:


for line in ([30.608, *line1], [22.810, *line2]):
    print('& {:.2f} & {:.2f} & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% \\\\'.format(*line))


# In[ ]:




