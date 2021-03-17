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
import numpy as np
from sklearn.model_selection import train_test_split
from os import path


# Hyperparameters
layers = 4
n_units = 400 # Number of neurons of the first 3 layers. 4th layer has 2 neurons
n_batch = 4096 # Batch size is the number of samples per gradient update.
n_epochs = 50


# Create DataFrame (df) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Processed data/options-df.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["bid_eod",
        "ask_eod"], axis = 1), call_df[["bid_eod", "ask_eod"]], test_size = 0.01)


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

history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)

model.save('Saved_models/mlp2_call_1')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_call_1_train_losses.txt", 
           numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_call_1_validation_losses.txt", 
           numpy_validation_loss, delimiter=",")

model.compile(loss='mse', optimizer=Adam(1e-4))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_call_2')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_call_2_train_losses.txt", 
           numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_call_2_validation_losses.txt", 
           numpy_validation_loss, delimiter=",")

model.compile(loss='mse', optimizer=Adam(1e-5))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_call_3')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_call_3_train_losses.txt", 
           numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_call_3_validation_losses.txt", 
           numpy_validation_loss, delimiter=",")

model.compile(loss='mse', optimizer=Adam(1e-6))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01, verbose=1)
model.save('Saved_models/mlp2_call_4')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_call_4_train_losses.txt", 
           numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_call_4_validation_losses.txt", 
           numpy_validation_loss, delimiter=",")

# SHORT TEST
# model.compile(loss='mse', optimizer=Adam(lr=1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                 batch_size=4096, epochs=2, validation_split = 0.01, verbose=1)
# model.save('mlp2_call_5')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp2_call_5_train_losses.txt", 
#            numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp2_call_5_validation_losses.txt", 
#            numpy_validation_loss, delimiter=",")

