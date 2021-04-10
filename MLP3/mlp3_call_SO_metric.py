#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:04:41 2021

@author: Diogo
"""

"""
Clear the console and remove all variables present on the namespace. This is 
useful to prevent Python to use more RAM each time I run the code.
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


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
call_df = pd.read_csv("call_df.csv")


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    call_df.drop(["Option_Average_Price"], 
    axis = 1), call_df.Option_Average_Price, test_size = 0.01))


# Create model using Keras' Functional API
# Create input layer
inputs = keras.Input(shape = (call_X_train.shape[1],))
x = layers.LeakyReLU(alpha = 1)(inputs)

"""
Function that creates a hidden layer by taking a tensor as input and applying a
modified ELU (MELU) activation function.
"""
def hl(tensor):
    # Create custom MELU activation function
    def melu(z):
        return tf.where(z > 0, ((z**2)/2 + 0.02*z) / (z - 2 + 1/0.49), 
                        0.49*(keras.activations.exponential(z)-1))
   
    y = layers.Dense(n_units, activation = melu)(tensor)
    return y

# Create hidden layers
for _ in range(n_hidden_layers):
    x = hl(x)

# Create output layer
outputs = layers.Dense(1, activation = keras.activations.softplus)(x)

# Actually create the model
model = keras.Model(inputs=inputs, outputs=outputs)


"""
Penalization function used in the metric for arbitrage-free prices
"""
def pen(x, lamb, m):
    return tf.where(x < 0, 0.0, lamb * x**m)

# Measure/Metric for the amount of prices that are not arbitrage-free.
def measure_arbitrage(y_true, y_pred):
    
    # Parameters of penalization function
    lamb = 1
    m = 0
    
    # Metric
    return (pen(-(model.input[:,0])**2 * tf.gradients(tf.gradients(y_pred, 
                    model.input), model.input)[0][:, 0], lamb, m) # constraint 1
            
            + pen(-model.input[:,1] * tf.gradients(y_pred, 
                    model.input)[0][:, 1], lamb, m) # constraint 2

            + pen(model.input[:,0] * tf.gradients(y_pred, 
                    model.input)[0][:, 0], lamb, m)) # constraint 3

# # Measure/Metric for the amount of prices that are not arbitrage-free.
# # Create variable that will store the metric
# arb = 0

# def measure_arbitrage(y_true, y_pred):
    
#     # Parameters of penalization function
#     lamb = 1
#     m = 0
    
#     # Metric
#     global arb
#     arb = arb + (pen(-(model.input[:,0])**2 * tf.gradients(tf.gradients(y_pred, 
#                     model.input), model.input)[0][:, 0], lamb, m) # constraint 1
            
#             + pen(-model.input[:,1] * tf.gradients(y_pred, 
#                     model.input)[0][:, 1], lamb, m) # constraint 2

#             + pen(model.input[:,0] * tf.gradients(y_pred, 
#                     model.input)[0][:, 0], lamb, m)) # constraint 3
    
#     return arb


# QUICK TEST
model.compile(loss = "mse", optimizer = keras.optimizers.Adam(), 
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