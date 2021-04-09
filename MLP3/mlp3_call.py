#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:53:08 2021

@author: Diogo
"""

from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Hyperparameters
n_hidden_layers = 2 # Number of hidden layers.
n_units = 128 # Number of neurons of the hidden layers.
n_batch = 64 # Number of observations used per gradient update.
n_epochs = 30


# Create DataFrame (df) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options-df.csv"))
df = pd.read_csv(filepath)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
# call_df = call_df.dropna()

# call_df.to_csv('call_df.csv', index=False)
# call_df = pd.read_csv("call_df.csv")


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(["Option_Average_Price"],
                    axis = 1), call_df.Option_Average_Price, test_size = 0.01)


# Create model using Keras' Functional API
# Create input layer
inputs = keras.Input(shape = (call_X_train.shape[1],))
x = layers.LeakyReLU(alpha = 1)(inputs)

# """
# Function that creates a hidden layer by taking a tensor as input and applying a
# modified ELU (MELU) activation function.
# """
# def hl(tensor):
#     # Create custom MELU activation function
#     def melu(z):
#         return tf.where(z > 0, ((z**2)/2 + 0.02*z) / (z - 2 + 1/0.49), 
#                         0.49*(keras.activations.exponential(z)-1))
    
#     # def melu(z):
#     #     return tf.cond(z > 0, lambda: ((z**2)/2 + 0.02*z) / (z - 2 + 1/0.49), 
#     #                     lambda: 0.49*(keras.activations.exponential(z)-1))
    
#     # from keras.utils.generic_utils import get_custom_objects
#     # get_custom_objects().update({'melu': layers.Activation(melu)})
    
#     # @tf.function
#     # def melu(z):
#     #     if z > 0:
#     #         return ((z**2)/2 + 0.02*z) / (z - 2 + 1/0.49)
#     #     else:
#     #         return 0.49*(keras.activations.exponential(z)-1)
    
#     y = layers.Dense(n_units, activation = melu)(tensor)
#     """
#     Dense() creates a densely-connected NN layer, implementing the following 
#     operation: output = activation(dot_product(input, kernel) + bias) where 
#     activation is the element-wise activation function passed as the activation
#     argument, kernel is a weights matrix created by the layer, and bias is a 
#     bias vector created by the layer (only applicable if use_bias is True, 
#     which it is by default). In this case we use the MELU activation function.
#     """
#     # x = layers.Dense(n_units)(tensor)
#     # y = layers.Activation(melu)(x)
#     return y

# hl function that works. Useful for tests.
def hl(tensor):
    lr = layers.Dense(n_units, activation = layers.LeakyReLU())(tensor)
    return lr

# Create hidden layers
for _ in range(n_hidden_layers):
    x = hl(x)

# Create output layer
outputs = layers.Dense(1, activation = keras.activations.softplus)(x)

# Actually create the model
model = keras.Model(inputs=inputs, outputs=outputs)


"""
Penalization function used in both the Custom loss funcion and the metric for 
arbitrage-free prices
"""
def pen(x, lamb, m):
    return tf.where(x < 0, 0.0, lamb * x**m)


# Custom loss function that is a MSE function plus three soft constraints
def constrained_mse(y_true, y_pred):
    
    # Parameters of penalization function
    lamb = 10
    m = 4
    
    # # Penalization function
    # # Parameters of said function
    # lamb = 10
    # m = 4
    
    # # Function
    # def pen(x):
    #     return tf.where(x < 0, 0.0, lamb * x**m)
    # def pen(x):
    #     return tf.cond(x < 0, lambda: 0.0, lambda: lamb * x**m)
    
    # Custom loss function
    return (keras.backend.mean(keras.backend.square(y_pred - y_true)) # MSE
            
            + pen(-(model.input[:,0])**2 * tf.gradients(tf.gradients(y_pred, 
                    model.input), model.input)[0][:, 0], lamb, m) # constraint 1
            
            + pen(-model.input[:,1] * tf.gradients(y_pred, 
                    model.input)[0][:, 1], lamb, m) # constraint 2

            + pen(model.input[:,0] * tf.gradients(y_pred, 
                    model.input)[0][:, 0], lamb, m)) # constraint 3


# Measure/Metric for the amount of prices that are not arbitrage-free.
def measure_arbitrage(y_true, y_pred):
    
    # Parameters of penalization function
    lamb = 1
    m = 0
    
    # # Function
    # def pen(x):
    #     return tf.cond(x < 0, lambda: 0.0, lambda: lamb * x**m)
    
    return (pen(-(model.input[:,0])**2 * tf.gradients(tf.gradients(y_pred, 
                    model.input), model.input)[0][:, 0], lamb, m) # constraint 1
            
            + pen(-model.input[:,1] * tf.gradients(y_pred, 
                    model.input)[0][:, 1], lamb, m) # constraint 2

            + pen(model.input[:,0] * tf.gradients(y_pred, 
                    model.input)[0][:, 0], lamb, m)) # constraint 3


# """
# Configure the learning process of the model with a loss function and an 
# optimizer. The optimizer changes the weights in order to minimize the loss 
# function. In this case the Adam optimizer will use the default learning rate 
# (LR) of 1e-3.
# """
# model.compile(loss = constrained_mse, optimizer = keras.optimizers.Adam(), 
#               metrics = [measure_arbitrage])
# # model.summary()

# """
# Train the model with batch_size = n_batch. See fit() method's arguments: 
# https://faroit.com/keras-docs/2.0.2/models/sequential/
# """
# history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
#                     epochs = n_epochs, validation_split = 0.01, verbose = 1)

# # Save the model's architecture, weights and optimizer's state
# model.save('Saved_models/mlp3_call_1')

# # Save the model's train and validation losses for each epoch.
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp3_call_1_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp3_call_1_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-4, batch size = 4096, epochs = n_epochs
# model.compile(loss = constrained_mse, 
#               optimizer = keras.optimizers.Adam(lr=1e-4), 
#               metrics = [measure_arbitrage])
# history = model.fit(call_X_train, call_y_train, batch_size = 4096, 
#                     epochs = n_epochs, validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp3_call_2')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp3_call_2_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp3_call_2_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-5, batch size = 4096, epochs = 10
# model.compile(loss = constrained_mse, 
#               optimizer = keras.optimizers.Adam(lr=1e-5), 
#               metrics = [measure_arbitrage])
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp3_call_3')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp3_call_3_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp3_call_3_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-6, batch size = 4096, epochs = 10
# model.compile(loss = constrained_mse, 
#               optimizer = keras.optimizers.Adam(lr=1e-6), 
#               metrics = [measure_arbitrage])
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp3_call_4')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp3_call_4_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp3_call_4_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# QUICK TEST
# model.compile(loss = "mse", optimizer = keras.optimizers.Adam())
# model.compile(loss = constrained_mse, optimizer = keras.optimizers.Adam())
model.compile(loss = constrained_mse, optimizer = keras.optimizers.Adam(), 
                metrics = [measure_arbitrage])

history = model.fit(call_X_train, call_y_train, 
                    batch_size = 4096, epochs = 20,
                    validation_split = 0.01, verbose = 1)
model.save('Saved_models/mlp3_call_test')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp3_call_test_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp3_call_test_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")

