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


# from keras.models import Sequential
# from keras.layers import Dense, LeakyReLU, BatchNormalization
# from keras.optimizers import Adam
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from os import path
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import math


# Hyper-parameters
n_hidden_layers = 5
# n_hidden_layers = 3
# n_hidden_layers = 1
n_units = 600 # Number of neurons of the hidden layers.
# n_units = 400
# n_units = 32
n_batch = 9216 # Number of observations used per gradient update.
# n_batch = 1024
# n_batch = 128
n_epochs = 40


"""Create DataFrame (DF) for calls"""
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                    "Processed data/options_phase3_final.csv"))                      
# filepath = path.abspath(path.join(basepath, "..", 
#                                   "Processed data/options_free_dataset.csv"))
df = pd.read_csv(filepath)

# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)

# basepath = path.dirname(__file__)
# filepath = path.abspath(path.join(basepath, "calls_prof.csv"))
# calls_df = pd.read_csv(filepath)

# """Remove first 100k observations because many of the variables show a weird 
# behavior in those."""
# calls_df = calls_df.iloc[100000:, :]


# Rescaling of the data
# calls_df["strike"] = minmax_scale(calls_df["strike"])
# calls_df["Option_Average_Price"] = minmax_scale(calls_df["Option_Average_Price"])
# calls_df["Underlying_Price"] = minmax_scale(calls_df["Underlying_Price"])
# calls_df["strike"] = robust_scale(calls_df["strike"])
# calls_df["Option_Average_Price"] = robust_scale(calls_df["Option_Average_Price"])
# calls_df["Underlying_Price"] = robust_scale(calls_df["Underlying_Price"])

# calls_df = StandardScaler().fit_transform(calls_df) 
# calls_df = pd.DataFrame(calls_df, columns = ['strike', 'Time_to_Maturity', 
#     'Option_Average_Price', 'RF_Rate', 'Sigma_20_Days_Annualized', 
#     'Underlying_Price']) 


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01))


# """
# Data normalization (subtract mean and divide by standard deviation)
# """
# def normalize(X_train, X_test):
# # def normalize(X_train, X_test, Y_train, Y_test):
#     X_train_mean = np.mean(X_train)
#     X_test_mean = np.mean(X_test)
#     # Y_train_mean = np.mean(Y_train)
#     # Y_test_mean = np.mean(Y_test)
#     X_train_std = np.std(X_train)
#     X_test_std = np.std(X_test)
#     # Y_train_std = np.std(Y_train)
#     # Y_test_std = np.std(Y_test)
#     X_train = (X_train - X_train_mean) / X_train_std
#     X_test = (X_test - X_test_mean) / X_test_std
#     # Y_train = (Y_train - Y_train_mean) / Y_train_std
#     # Y_test  = (Y_test - Y_test_mean) / Y_test_std
#     return X_train, X_test
#     # return X_train, X_test, Y_train, Y_test

# call_X_train, call_X_test = normalize(call_X_train, call_X_test)
# # call_X_train, call_X_test, call_y_train, call_y_test = normalize(call_X_train, 
# #                                         call_X_test, call_y_train, call_y_test)
# print("Are there any 'nan' values in the training sample?", np.any(np.isnan(
#                                                                 call_X_train)))


# # Normalize the inputs only
# def normalize(x_train, x_test):
#     train_mean = np.mean(x_train)
#     train_std = np.mean(x_train)
#     x_train = (x_train - train_mean)/train_std
#     x_test = (x_test - train_mean)/train_std
#     return x_train, x_test

# call_X_train, call_X_test = normalize(call_X_train, call_X_test)
# print(np.any(np.isnan(call_X_train)))


# Create model using Keras' functional API
# Create input layer
inputs = keras.Input(shape = (call_X_train.shape[1],))
x = layers.LeakyReLU()(inputs)
# x = layers.LeakyReLU(0.1)(inputs)

"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor):
    # initializer = tf.keras.initializers.GlorotUniform() 
    # initializer = tf.keras.initializers.Constant()
    # initializer = tf.keras.initializers.he_normal()
    # initializer = tf.keras.initializers.RandomNormal(
        # stddev = math.sqrt(4 / (32 + 32)))
    dense_layer = layers.Dense(n_units)
    # dense = layers.Dense(n_units, kernel_initializer = initializer, 
                # kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2 = 1e-4))
    # dense = layers.Dense(n_units, kernel_initializer = initializer,
    #                                   bias_initializer = initializer)
    
    
    """Dense() creates a densely-connected NN layer, implementing the following 
        operation: output = activation(dot_product(input, kernel) + bias) 
        where activation is the element-wise activation function passed as the 
        activation argument, kernel is a weights matrix created by the layer, 
        and bias is a bias vector created by the layer (only applicable if 
        use_bias is True, which it is by default). In this case no activation 
        function was passed so there is "linear" activation: a(x) = x."""
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
    x = hl(x)

# Create output layer
outputs = layers.Dense(1, activation='relu')(x)

# Actually create the model
model = keras.Model(inputs = inputs, outputs = outputs)


# # Create a Sequential model that is a linear stack of layers
# model = Sequential()

# # Add layers incrementally
# # Create input layer
# model.add(Dense(n_units, input_dim = call_X_train.shape[1])) 
# # Dense() creates a densely-connected NN layer, implementing the following 
#     # operation: output = activation(dot_product(input, kernel) + bias) where 
#     # activation is the element-wise activation function passed as the 
#     # activation argument, kernel is a weights matrix created by the layer, 
#     # and bias is a bias vector created by the layer (only applicable if 
#     # use_bias is True, which it is by default). In this case no activation 
#     # function was passed so there is "linear" activation: a(x) = x.
# # The first parameter in Dense() sets the number of neurons of that layer. In 
#     # this case = n_units.
# # The second parameter, input_dim, defines how many inputs the layer is going 
#     # to have. In this case it's 5 = put_X_train.shape[1] = strike price, time 
#     # to maturity, risk-free rate, historical volatility and the price of the
#     # underlying asset.
# model.add(LeakyReLU())

# # Create hidden layers
# for _ in range(layers - 1):
#     model.add(Dense(n_units)) 
#     model.add(BatchNormalization()) # Batch normalization scales the output of 
#         # a layer by subtracting the batch mean and dividing by the batch 
#         # standard deviation (so it maintains the output's mean close to 0 and 
#         # it's standard deviation close to 1). This can speed up the training 
#         # of the neural network.
#     model.add(LeakyReLU())

# # Create output layer
# model.add(Dense(1, activation='relu'))


# """Configure the learning process, train the model, save model and it's 
# losses, with different learning rates, batch sizes and number of epochs."""
"""Configure the learning process, train the model, and save the model and it's 
losses"""

"""Configure the learning process of the model with a loss function and an 
optimizer. The optimizer changes the weights in order to minimize the loss 
function. In this case we use the Adam optimizer"""
model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
    
"""Train the model, given certain hyper-parameters"""
history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
                    epochs = n_epochs, validation_split = 0.01, verbose = 1)

# # Configure the learning process of the model with a loss function and an 
#     # optimizer. The optimizer changes the weights in order to minimize the 
#     # loss function. In this case the Adam optimizer will use the default 
#     # learning rate (LR) of 1e-3.
# model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3))
# # model.summary()

# # Train the model with batch_size = n_batch. See fit() method's arguments: 
#     # https://faroit.com/keras-docs/2.0.2/models/sequential/
# history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
#                     epochs = n_epochs, validation_split = 0.01, verbose = 1)

# Save the model's architecture, weights and optimizer's state
model.save('Saved_models/mlp1_call_1')

# Save the model's train and validation losses for each epoch.
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp1_call_1_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp1_call_1_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")


# Another network using the hyper-parameters in Ke and Yang (2019)
# LR changes with the number of epochs, batch size = 4096, epochs = 30
step = tf.Variable(0, trainable=False)
boundaries = [10, 20]
values = [1e-3, 1e-4, 1e-5]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

learning_rate = learning_rate_fn(step)

model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr = learning_rate))
history = model.fit(call_X_train, call_y_train, batch_size=4096, 
                    epochs = 30, validation_split = 0.01, verbose = 1)
model.save('Saved_models/mlp1_call_2')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp1_call_2_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp1_call_2_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")


# # LR = 1e-4, batch size = 4096, epochs = n_epochs
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-4))
# history = model.fit(call_X_train, call_y_train, batch_size=4096, 
#                     epochs=n_epochs, validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_call_2')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_call_2_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_call_2_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-5, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-5))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_call_3')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_call_3_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_call_3_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-6, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-6))
# history = model.fit(call_X_train, call_y_train, 
#                     batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_call_4')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_call_4_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_call_4_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")


# # QUICK TEST
# # model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr = 1e-3))

# # initial_lr = 1e-1
# # global_step = tf.Variable(0, trainable=False)
# # decayed_lr = tf.compat.v1.train.exponential_decay(starter_lr, global_step,
# #                                                  10000, 0.95, staircase = True)

# # decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, 
# #         decay_steps = 100000, decay_rate = 0.96, staircase = True)

# # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(
# #                                                 learning_rate = decayed_lr))

# model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
# # history = model.fit(call_X_train, call_y_train, batch_size = 4096, epochs = 1, 
#                     # validation_split = 0.01, verbose = 1)
# history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
#                     epochs = 1, validation_split = 0.01, verbose = 1)
# model.save('Saved_models/mlp1_call_test')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_call_test_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_call_test_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

