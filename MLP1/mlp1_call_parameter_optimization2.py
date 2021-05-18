# -*- coding: utf-8 -*-
"""
Created on Tue May 18 08:48:22 2021

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


import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import talos


# # Hyperparameters
# n_hidden_layers = 3
# n_units = 400 # Number of neurons of the hidden layers.
# n_batch = 1024 # Number of observations used per gradient update.
# n_epochs = 40


# Create DataFrame (DF) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
frames = [calls_df.iloc[:, :2], calls_df.iloc[:, 3:], calls_df.iloc[:, 2]]
calls_df = pd.concat([frames[0], frames[1]], axis = 1)
calls_df = pd.concat([calls_df, frames[2]], axis = 1)
x = calls_df.iloc[:, 0:5]
y = calls_df.iloc[:, 5]

# """Split call_df into random train and test subsets, for inputs (X) and 
# output (y)"""
# call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
#     calls_df.drop(["Option_Average_Price"], axis = 1), 
#     calls_df.Option_Average_Price, test_size = 0.01))


"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor, n_units):
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
    """
    Batch normalization scales the output of a layer by subtracting the batch
    mean and dividing by the batch standard deviation (so the output's mean 
    will be close to 0 and it's standard deviation close to 1). Theoretically 
    this can speed up the training of the neural network.
    """
    lr = layers.LeakyReLU()(bn)
    return lr

# Create MLP1 model using Keras' functional API
def make_model(call_X_train, call_y_train, params):
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(params["n_hidden_layers"]):
        x = hl(x, params["n_units"])
        
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3))
    model.compile(loss = 'mse', 
       optimizer = keras.optimizers.Adam(lr = params["learning_rate"]))
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(), 
    #               metrics=["accuracy"]) # added metrics to see less warnings
    
    # Train the model
    out = model.fit(call_X_train, call_y_train, 
                    batch_size = params["n_batch"], 
                    epochs = params["n_epochs"], 
                    validation_split = 0.01, verbose = 1)
    return out, model


parameter_grid = dict(n_hidden_layers = np.arange(1, 11, 1).tolist(), 
                      n_units = np.arange(100, 1001, 100).tolist(), 
                      n_batch = np.arange(1024, 10241, 1024).tolist(), 
                      n_epochs = [10, 20, 30, 41], 
                      learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

t = talos.Scan(x = x, y = y, params = parameter_grid, 
               model = make_model, experiment_name = 'MLP1_calls')

