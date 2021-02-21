#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:50:11 2021

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


# Hyperparameters
layers = 4
n_units = 400 # Number of neurons of the first 3 layers. 4th layer has 1 neuron.
n_batch = 1024 # Batch size is the number of samples per gradient update.
n_epochs = 40


# Create DataFrame (df) for calls
df = pd.read_csv("options-df.csv")
# df = df.dropna(axis=0)
df = df.drop(columns=['Bid', 'Ask', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
call_df = df[df.OptionType == 'call'].drop(['OptionType'], axis=1)


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Average_Price"],
                            axis = 1), call_df.Average_Price, test_size = 0.01)


# Create a Sequential model that is a linear stack of layers
model = Sequential()

# Adds layers incrementally
model.add(Dense(n_units, input_dim = call_X_train.shape[1])) # The parameter
    # input_dim defines how many inputs the layer is going to have. In this
    # case 5 = call_X_train.shape[1].
    # The parameter units (in this case = n_units) sets the dimensionality of 
        # the output space.
model.add(LeakyReLU())

for _ in range(layers - 1):
    model.add(Dense(n_units)) # Dense() creates a densely-connected NN layer, 
        # implementing the following operation: output = activation(dot(input, 
        # kernel) + bias) where activation is the element-wise activation 
        # function passed as the activation argument, kernel is a weights 
        # matrix created by the layer, and bias is a bias vector created by the 
        # layer (only applicable if use_bias is True, which it is by default). 
        # In this case no activation function was passed so there is "linear" 
        # activation: a(x) = x.
    model.add(BatchNormalization()) # Batch normalization scales the output of 
        # a layer by subtracting the batch mean and dividing by the batch 
        # standard deviation (so it maintains the output's mean close to 0 and 
        # it's standard deviation close to 1. This can speed up the training of 
        # the neural network.
    model.add(LeakyReLU())

model.add(Dense(1, activation='relu')) # Define output layer?


# # Configure the learning process, train the model and save model, with 
#     # different learning rates, batch sizes number of epochs.
    
# # Configure the learning process of the model with a loss function and an 
#     # optimizer. The optimizer changes the weights in order to minimize the 
#     # loss function. In this case the Adam optimizer will use the default 
#     # learning rate (LR) of 1e-3
# model.compile(loss = 'mse', optimizer = Adam())
# # model.summary()
# # len(model.layers) # number of layers in the model/network

# # Train the model with batch_size = n_batch. See fit() method's arguments: 
#     # https://faroit.com/keras-docs/2.0.2/models/sequential/
# history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
#                     epochs = n_epochs, validation_split = 0.01, verbose = 1)

# # Save the model's configuration, weights and optimizer's state


# # LR = 1e-4, batch size = 4096, epochs = n_epochs
# model.compile(loss='mse', optimizer=Adam(lr=1e-4))
# history = model.fit(call_X_train, call_y_train, batch_size=4096, 
#                     epochs=n_epochs, validation_split = 0.01, verbose=1)
# model.save('mlp1_call_2')

# # LR = 1e-5, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer=Adam(lr=1e-5))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, validation_split = 0.01, verbose=1)
# model.save('mlp1_call_3')

# # LR = 1e-6, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer=Adam(lr=1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('mlp1_call_4')

# SHORT TEST
model.compile(loss='mse', optimizer=Adam(lr=1e-6))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=4096, epochs=2, validation_split = 0.01, verbose=1)
model.save('mlp1_call_5')


# # Generate output predictions for the input test sample
# call_y_pred = model.predict(call_X_test)

# # Differences between the output's test values and the output's predictions
# diff = (call_y_test.values - call_y_pred.reshape(call_y_pred.shape[0]))

# # Mean of the square of the differences (MSE)
# np.mean(np.square(diff))

