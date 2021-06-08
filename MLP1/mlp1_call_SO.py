# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:14:43 2021

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
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from os import path


# Hyper-parameters
n_hidden_layers = 3
n_units = 400 # Number of neurons of the hidden layers.
# n_batch = 1024 # Number of observations used per gradient update.
# n_epochs = 40


"""Create DataFrame (DF) for calls"""
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_free_dataset.csv"))
df = pd.read_csv(filepath)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)

"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01))


"""
Data normalization according to chapter 7.6 of Varma and Das (2018)
"""
def normalize(X_train, X_test):
# def normalize(X_train, X_test, Y_train, Y_test):
    X_train_mean = np.mean(X_train)
    X_test_mean = np.mean(X_test)
    # Y_train_mean = np.mean(Y_train)
    # Y_test_mean = np.mean(Y_test)
    X_train_std = np.std(X_train)
    X_test_std = np.std(X_test)
    # Y_train_std = np.std(Y_train)
    # Y_test_std = np.std(Y_test)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_test_mean) / X_test_std
    # Y_train = (Y_train - Y_train_mean) / Y_train_std
    # Y_test  = (Y_test - Y_test_mean) / Y_test_std
    return X_train, X_test
    # return X_train, X_test, Y_train, Y_test

call_X_train, call_X_test = normalize(call_X_train, call_X_test)
# call_X_train, call_X_test, call_y_train, call_y_test = normalize(call_X_train, 
#                                         call_X_test, call_y_train, call_y_test)
print("Are there any 'nan' values in the training sample?", 
      np.any(np.isnan(call_X_train)))


"""
Create model using Keras' functional API
"""
# Create input layer
inputs = keras.Input(shape = (call_X_train.shape[1],))
x = layers.LeakyReLU()(inputs)

"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor):
    dense = layers.Dense(n_units)
    x = dense(tensor)
    bn = layers.BatchNormalization()(x)
    leaky = layers.LeakyReLU()(bn)
    return leaky

# Create hidden layers
for _ in range(n_hidden_layers):
    x = hl(x)


# """
# Create hidden layers without a loop in order to initialize the parameters
# according o the suggestions in chapter 7.5 of Varma and Das (2018)
# """
# # r = math.sqrt(12 / (5 + n_units))
# # initializer = tf.keras.initializers.RandomUniform(minval = -r, maxval = r)
# initializer = tf.keras.initializers.RandomNormal(
#                 stddev = math.sqrt(4 / (5 + n_units)))
# dense = layers.Dense(n_units, kernel_initializer = initializer)
# x = dense(x)
# leaky = layers.LeakyReLU()(x)

# # r = math.sqrt(12 / (n_units + n_units))
# # initializer = tf.keras.initializers.RandomUniform(minval = -r, maxval = r)
# initializer = tf.keras.initializers.RandomNormal(
#                 stddev = math.sqrt(4 / (n_units + n_units)))
# dense = layers.Dense(n_units, kernel_initializer = initializer)
# x = dense(leaky)
# leaky = layers.LeakyReLU()(x)

# # r = math.sqrt(12 / (n_units + 1))
# # initializer = tf.keras.initializers.RandomUniform(minval = -r, maxval = r)
# initializer = tf.keras.initializers.RandomNormal(
#                 stddev = math.sqrt(4 / (n_units + 1)))
# dense = layers.Dense(n_units, kernel_initializer = initializer)
# x = dense(leaky)
# leaky = layers.LeakyReLU()(x)


# Create output layer
outputs = layers.Dense(1, activation='relu')(x)

# Actually create the model
model = keras.Model(inputs = inputs, outputs = outputs)


"""
Configure the learning process, train the model, and save the model and it's 
losses
"""
model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
history = model.fit(call_X_train, call_y_train, batch_size = 4096, epochs = 1, 
                    validation_split = 0.01, verbose = 1)
model.save('Saved_models/mlp1_call_test')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp1_call_test_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp1_call_test_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")

