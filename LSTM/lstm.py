#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:05:11 2021

@author: Diogo
"""

# from keras.models import Sequential, Model, load_model
from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, LSTM, Bidirectional, Input, Concatenate
from keras.layers import Dense, LeakyReLU, BatchNormalization, LSTM, Input, Concatenate
# from keras import backend as K
# from keras.callbacks import TensorBoard
from keras.optimizers import Adam
# from keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os import path


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "options-df.csv"))
options_df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
options_df = options_df.drop(columns=['Sigma_20_Days', "Underlying_Price", 
                                      "bid_eod", "ask_eod"])
filepath = path.abspath(path.join(basepath, "..", "underlying_df.csv"))
underlying = pd.read_csv(filepath)


# Hyperparameters
layers = 4
n_timesteps = 60
features = 4
n_batch = 4096
n_epochs = 100
N_TIMESTEPS = 20


# Create array with the prices of the underlying, where the first N_TIMESTEPS 
    # entries are nan
padded = np.insert(underlying[" Close"].values, 0, np.array([np.nan] * N_TIMESTEPS))

# Create Dataframe (df) where each row has N_TIMESTEPS prices of the underlying, 
    # ordered by date, in descending order. From one row to the next each price
    # is replaced by the one observed in the next date.
rolled = np.column_stack([np.roll(padded, i) for i in range(N_TIMESTEPS)])
rolled = rolled[~np.isnan(rolled).any(axis=1)]
rolled = np.column_stack((underlying.Date.values[N_TIMESTEPS - 1:], rolled))
price_history = pd.DataFrame(data=rolled)

# Add columns of price_history to df, according to the date in "QuoteDate"
joined = options_df.join(price_history.set_index(0), on='QuoteDate')

# Create dfs from calls and puts
call_df = joined[joined.OptionType == 'c'].drop(['OptionType'], axis=1)
call_df = call_df.drop(columns=['QuoteDate'])
put_df = joined[joined.OptionType == 'p'].drop(['OptionType'], axis=1)
put_df = put_df.drop(columns=['QuoteDate'])

#
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Option_Average_Price"], 
            axis=1).values, call_df.Option_Average_Price.values, test_size=0.01)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(["Option_Average_Price"], 
            axis=1).values, put_df.Option_Average_Price.values, test_size=0.01)

# 
call_X_train = [call_X_train[:, -N_TIMESTEPS:].reshape(call_X_train.shape[0], 
                                        N_TIMESTEPS, 1), call_X_train[:, :4]]
call_X_test = [call_X_test[:, -N_TIMESTEPS:].reshape(call_X_test.shape[0], 
                                        N_TIMESTEPS, 1), call_X_test[:, :4]]
put_X_train = [put_X_train[:, -N_TIMESTEPS:].reshape(put_X_train.shape[0], 
                                        N_TIMESTEPS, 1), put_X_train[:, :4]]
put_X_test = [put_X_test[:, -N_TIMESTEPS:].reshape(put_X_test.shape[0], 
                                        N_TIMESTEPS, 1), put_X_test[:, :4]]


# 
def make_model():
    close_history = Input((N_TIMESTEPS, 1))
        
    lstm = Sequential()
    lstm.add(LSTM(units=8, input_shape=(N_TIMESTEPS, 1), return_sequences=True))
    lstm.add(LSTM(units=8, return_sequences=True))
    lstm.add(LSTM(units=8, return_sequences=True))
    lstm.add(LSTM(units=8, return_sequences=False))
    input1 = lstm(close_history)
    
    input2 = Input((features,))
    connect = Concatenate()([input1, input2])
    
    for _ in range(layers - 1):
        connect = Dense(100)(connect)
        connect = BatchNormalization()(connect)
        connect = LeakyReLU()(connect)
    
    predict = Dense(1, activation='relu')(connect)

    return Model(inputs=[close_history, input2], outputs=predict)

# 
call_model = make_model()
put_model = make_model()
# call_model.summary()

call_model.compile(optimizer=Adam(lr=1e-2), loss='mse')
history = call_model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01, verbose=1)
call_model.save('Saved_models/lstm_call_1')

call_model.compile(optimizer=Adam(lr=1e-3), loss='mse')
history = call_model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01, verbose=1)
call_model.save('Saved_models/lstm_call_2')

call_model.compile(optimizer=Adam(lr=1e-4), loss='mse')
history = call_model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01, verbose=1)
call_model.save('Saved_models/lstm_call_3')

put_model.compile(optimizer=Adam(lr=1e-2), loss='mse')
history = put_model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01, verbose=1)
put_model.save('Saved_models/lstm_put_1')

put_model.compile(optimizer=Adam(lr=1e-3), loss='mse')
history = put_model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01, verbose=1)
put_model.save('Saved_models/lstm_put_2')

put_model.compile(optimizer=Adam(lr=1e-4), loss='mse')
history = put_model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01, verbose=1)
put_model.save("Saved_models/lstm_put_3")

# SHORT TEST
# call_model.compile(loss='mse', optimizer=Adam(lr=1e-6))
# history = call_model.fit(call_X_train, call_y_train, 
#                 batch_size=4096, epochs=2, validation_split = 0.01, verbose=1)
# call_model.save('Saved_models/lstm_call_5')

