#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:29:12 2021

@author: Diogo
"""

# from keras.models import Sequential
# from keras.layers import Dense, LeakyReLU, BatchNormalization
# from keras.optimizers import Adam
import pandas as pd
import numpy as np
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from os import path


# Hyperparameters
n_hidden_layers = 3
n_units = 400 # Number of neurons of the hidden layers.
n_batch = 4096 # Number of observations used per gradient update.
n_epochs = 30


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


# Create model using Keras' functional API
# Create input layer
inputs = keras.Input(shape = (call_X_train.shape[1],))
x = layers.LeakyReLU()(inputs)

# Create function that creates a hidden layer by taking a tensor as input and 
    # applying Batch Normalization and the LeakyReLU activation.
def hl(tensor):
    dense = layers.Dense(n_units)
    # Dense() creates a densely-connected NN layer, implementing the following 
        # operation: output = activation(dot_product(input, kernel) + bias) 
        # where activation is the element-wise activation function passed as the 
        # activation argument, kernel is a weights matrix created by the layer, 
        # and bias is a bias vector created by the layer (only applicable if 
        # use_bias is True, which it is by default). In this case no activation 
        # function was passed so there is "linear" activation: a(x) = x.
    x = dense(tensor)
    bn = layers.BatchNormalization()(x)
    # Batch normalization scales the output of a layer by subtracting the batch
        # mean and dividing by the batch standard deviation (so it maintains 
        # the output's mean close to 0 and it's standard deviation close to 1).
        # Theoretically this can speed up the training of the neural network.
    lr = layers.LeakyReLU()(bn)
    return lr

# Create hidden layers
for _ in range(n_hidden_layers):
    x = hl(x)

# Create output layer
outputs = layers.Dense(2, activation='relu')(x)

# Actually create the model
model = keras.Model(inputs=inputs, outputs=outputs)


# # Create a Sequential model that is a linear stack of layers
# model = Sequential()

# # Adds layers incrementally
# model.add(Dense(n_units, input_dim=call_X_train.shape[1]))
# model.add(LeakyReLU())

# for _ in range(layers - 1):
#     model.add(Dense(n_units))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())

# model.add(Dense(2, activation='relu'))


# Configure the learning process, train the model, save model and it's losses, 
    # with different learning rates, batch sizes and number of epochs.
model.compile(loss='mse', optimizer = keras.optimizers.Adam())
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

model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-4))
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

# model.compile(loss='mse', optimizer = keras.optimizers.Adam(1e-5))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=n_batch, epochs=n_epochs, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp2_call_3')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp2_call_3_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp2_call_3_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# model.compile(loss='mse', optimizer = keras.optimizers.Adam(1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=n_batch, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp2_call_4')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp2_call_4_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp2_call_4_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # SHORT TEST
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                 batch_size=4096, epochs=1, validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp2_call_5')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp2_call_5_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp2_call_5_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

