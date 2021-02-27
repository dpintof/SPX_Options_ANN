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
from os import path


# Hyperparameters
layers = 4
n_units = 400 # Number of neurons of the first 3 layers. 4th layer has 2 neurons
n_batch = 4096 # Batch size is the number of samples per gradient update.
n_epochs = 50


# Create DataFrame (df) for puts
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "options-df.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
put_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


# Split put_df into random train and test subsets, for inputs (X) and output (y)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(["bid_eod",
        "ask_eod"], axis = 1), put_df[["bid_eod", "ask_eod"]], test_size = 0.01)


# Create a Sequential model that is a linear stack of layers
model = Sequential()

# Adds layers incrementally
model.add(Dense(n_units, input_dim=put_X_train.shape[1]))
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

history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)

model.save('Saved_models/mlp2_put_1')

model.compile(loss='mse', optimizer=Adam(1e-4))
history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_put_2')

model.compile(loss='mse', optimizer=Adam(1e-5))
history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_put_3')

model.compile(loss='mse', optimizer=Adam(1e-6))
history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_put_4')

# SHORT TEST
# model.compile(loss='mse', optimizer=Adam(lr=1e-6))
# history = model.fit(put_X_train, put_y_train, 
#                 batch_size=4096, epochs=2, validation_split = 0.01, verbose=1)
# model.save('mlp2_put_5')
