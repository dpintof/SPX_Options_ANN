#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:29:12 2021

@author: Diogo
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from os import path


# Hyperparameters
n_hidden_layers = 4
n_units = 400 # Number of neurons of the hidden layers.
n_batch = 4096 # Number of observations used per gradient update.
n_epochs = 50
learning_rate = 0.001


# Create DataFrame (df) for puts
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
put_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


# Split put_df into random train and test subsets, for inputs (X) and output (y)
put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(
    put_df.drop(["bid_eod", "ask_eod"], axis = 1), 
    put_df[["bid_eod", "ask_eod"]], test_size = 0.01)


# Create model using Keras' functional API
def mlp2_put(n_units, n_hidden_layers):
    
    # Create input layer
    inputs = keras.Input(shape = (put_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)
    # x = layers.LeakyReLU(0.1)(inputs)

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
        # leaky = layers.LeakyReLU(0.1)(bn)
        return leaky

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x, n_units)
    
    # Create output layer
    outputs = layers.Dense(2, activation='relu')(x)
    
    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)

    return model


# CONFIGURE THE LEARNING PROCESS, TRAIN THE MODEL, SAVE MODEL AND IT'S LOSSES, 
# WITH DIFFERENT LEARNING RATES, BATCH SIZES AND NUMBER OF EPOCHS.
# Learning_rate = learning_rate as defined in the hyperparameters section
model = mlp2_put(n_units, n_hidden_layers)
model.compile(loss='mse', optimizer = keras.optimizers.Adam(
                                                learning_rate=learning_rate))
history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)

# Introduced "directory" to prevent an error when running the code on Windows
directory = path.join("Saved_models", "mlp2_put_1")
model.save(directory)

train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_put_1_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_put_1_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")


# Learning rate changes with the number of epochs, as in Ke and Yang (2019)
step = tf.Variable(0, trainable = False)
boundaries = [10, 20]
values = [1e-3, 1e-4, 1e-5]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

learning_rate = learning_rate_fn(step)

model = mlp2_put(n_units, n_hidden_layers)
model.compile(loss='mse', optimizer = keras.optimizers.Adam(
                                                learning_rate=learning_rate))
history = model.fit(put_X_train, put_y_train, 
                    batch_size=n_batch, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)
directory = path.join("Saved_models", "mlp2_put_2")
model.save(directory)
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp2_put_2_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp2_put_2_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")

