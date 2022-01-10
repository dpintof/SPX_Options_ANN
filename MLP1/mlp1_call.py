#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:50:11 2021

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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from os import path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import math


# Hyper-parameters ("paper" refers to Ke and Yang [2019])
n_hidden_layers = 5
n_hidden_layers_paper = 3
n_units = 600 # Number of neurons of the hidden layers.
n_units_paper = 300
n_batch = 9216 # Number of observations used per gradient update.
n_batch_paper = 4096
n_epochs = 35
n_epochs_paper = 30


"""Create DataFrame (DF) for calls"""
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                    "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)

df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01))


# Create model using Keras' functional API
def mlp1_call(n_units, n_hidden_layers):
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    """Function that creates a hidden layer by taking a tensor as input and 
    applying Batch Normalization and the LeakyReLU activation."""
    def hl(tensor, n_units):

        dense_layer = layers.Dense(n_units)
        """Dense() creates a densely-connected NN layer, implementing the 
        following operation: output = activation(dot_product(input, kernel) + 
        bias) where activation is the element-wise activation function passed 
        as the activation argument, kernel is a weights matrix created by the 
        layer, and bias is a bias vector created by the layer (only applicable 
        if use_bias is True, which it is by default). In this case no 
        activation function was passed so there is "linear" activation: a(x) = 
        x."""
        x = dense_layer(tensor)
        bn = layers.BatchNormalization()(x)
        """
        Batch normalization scales the output of a layer by subtracting the batch
        mean and dividing by the batch standard deviation (so the output's mean 
        will be close to 0 and it's standard deviation close to 1). Theoretically 
        this can speed up the training of the neural network.
        """
        leaky = layers.LeakyReLU()(bn)
        return leaky

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x, n_units)
    
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)
    
    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)

    return model


# CONFIGURE THE LEARNING PROCESS, TRAIN THE MODEL, SAVE MODEL AND IT'S 
# LOSSES, WITH DIFFERENT LEARNING RATES, BATCH SIZES AND NUMBER OF EPOCHS.
"""Model using the hyper-parameters in Ke and Yang (2019). LR changes with the 
number of epochs, batch size = 4096, epochs = 30"""
step = tf.Variable(0, trainable = False)
boundaries = [10, 20]
values = [1e-3, 1e-4, 1e-5]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

learning_rate = learning_rate_fn(step)

model = mlp1_call(n_units_paper, n_hidden_layers_paper) 
model.compile(loss='mse', optimizer = keras.optimizers.Adam(
                                        lr = learning_rate))
history = model.fit(call_X_train, call_y_train, batch_size = n_batch_paper, 
                    epochs = n_epochs_paper, validation_split = 0.01, 
                    verbose = 1)
model.save('Saved_models/mlp1_call_2')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp1_call_2_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp1_call_2_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")

