#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:29:12 2021

@author: Diogo
"""

from keras.models import Sequential
# from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Dense, LeakyReLU, BatchNormalization
# from keras import backend
# from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split


# Hyperparameters
layers = 4
n_units = 400 # Number of neurons of the first 3 layers. 4th layer has 2 neurons
n_batch = 4096 # Batch size is the number of samples per gradient update.
n_epochs = 50


# Create DataFrame (df) for calls
df = pd.read_csv("options-df.csv")
# df = df.dropna(axis=0)
df = df.drop(columns=['Average_Price', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
call_df = df[df.OptionType == 'call'].drop(['OptionType'], axis=1)


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Bid",
                    "Ask"], axis = 1), call_df[["Bid", "Ask"]], test_size = 0.01)


# Create a Sequential model that is a linear stack of layers
model = Sequential()

# Adds layers incrementally
model.add(Dense(n_units, input_dim=call_X_train.shape[1]))
model.add(LeakyReLU())

for _ in range(layers - 1):
    model.add(Dense(n_units))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

model.add(Dense(2, activation='relu'))


# Configure the learning process, train the model and save model, with 
    # different learning rates, batch sizes number of epochs.
model.compile(loss='mse', optimizer=Adam())

# model.summary()

history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)

model.save('mlp2_call_1')

# call_y_pred30 = model.predict(call_X_test)
# print('equilibrium mse', np.mean(np.square(np.mean(call_y_test.values, axis=1) - np.mean(call_y_pred30, axis=1))))
# print('spread mse', np.mean(np.square(np.diff(call_y_test.values, axis=1) - np.diff(call_y_pred30, axis=1))))

model.compile(loss='mse', optimizer=Adam(1e-4))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('mlp2_call_2')

# call_y_pred40 = model.predict(call_X_test)
# print('equilibrium mse', np.mean(np.square(np.mean(call_y_test.values, axis=1) - np.mean(call_y_pred40, axis=1))))
# print('spread mse', np.mean(np.square(np.diff(call_y_test.values, axis=1) - np.diff(call_y_pred40, axis=1))))

model.compile(loss='mse', optimizer=Adam(1e-5))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('mlp2_call_3')

# call_y_pred50 = model.predict(call_X_test)
# print('equilibrium mse', np.mean(np.square(np.mean(call_y_test.values, axis=1) - np.mean(call_y_pred50, axis=1))))
# print('spread mse', np.mean(np.square(np.diff(call_y_test.values, axis=1) - np.diff(call_y_pred50, axis=1))))

model.compile(loss='mse', optimizer=Adam(1e-6))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01, verbose=1)
model.save('mlp2_call_4')

# call_y_pred60 = model.predict(call_X_test)
# print('equilibrium mse', np.mean(np.square(np.mean(call_y_test.values, axis=1) - np.mean(call_y_pred60, axis=1))))
# print('spread mse', np.mean(np.square(np.diff(call_y_test.values, axis=1) - np.diff(call_y_pred60, axis=1))))

# SHORT TEST
# model.compile(loss='mse', optimizer=Adam(lr=1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                 batch_size=4096, epochs=2, validation_split = 0.01, verbose=1)
# model.save('mlp2_call_5')
