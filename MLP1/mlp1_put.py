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


# Hyperparameters
n_hidden_layers = 5
n_hidden_layers_paper = 3
n_units = 600 # Number of neurons of the hidden layers.
n_units_paper = 300
n_batch = 9216 # Number of observations used per gradient update.
n_batch_paper = 4096
n_epochs = 35
n_epochs_paper = 30


# Create DataFrame (df) for puts
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                    "Processed data/options_phase3_final.csv"))                      
# filepath = path.abspath(path.join(basepath, "..", 
#                                   "Processed data/options_free_dataset.csv"))
df = pd.read_csv(filepath)

df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
puts_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


"""Split both inputs (X) and output (y), from puts_df, into random train and test 
subsets"""
puts_X_train, puts_X_test, puts_y_train, puts_y_test = train_test_split(
    puts_df.drop(["Option_Average_Price"], axis = 1), 
    puts_df.Option_Average_Price, test_size = 0.01)


# Create model using Keras' functional API
def mlp1_put(n_neurons, n_hidden_layers):

    # Create input layer
    inputs = keras.Input(shape = (puts_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    """Create function that creates a hidden layer by taking a tensor as input 
    and applying Batch Normalization and the LeakyReLU activation."""
    def hl(tensor, n_neurons):
        dense = layers.Dense(n_neurons)
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
        leaky = layers.LeakyReLU()(bn)
        return leaky

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x, n_neurons)
    
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)
    
    # Actually create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# # Create a Sequential model that is a linear stack of layers
# model = Sequential()

# # Adds layers incrementally
# model.add(Dense(n_units, input_dim = put_X_train.shape[1])) # The parameter
#     # input_dim defines how many inputs the layer is going to have. In this
#     # case it's 5 = put_X_train.shape[1].
#     # The parameter units (in this case = n_units) sets the dimensionality of 
#         # the output space.
# model.add(LeakyReLU())

# for _ in range(layers - 1):
#     model.add(Dense(n_units)) # Dense() creates a densely-connected NN layer, 
#         # implementing the following operation: output = activation(dot(input, 
#         # kernel) + bias) where activation is the element-wise activation 
#         # function passed as the activation argument, kernel is a weights 
#         # matrix created by the layer, and bias is a bias vector created by the 
#         # layer (only applicable if use_bias is True, which it is by default). 
#         # In this case no activation function was passed so there is "linear" 
#         # activation: a(x) = x.
#     model.add(BatchNormalization()) # Batch normalization scales the output of 
#         # a layer by subtracting the batch mean and dividing by the batch 
#         # standard deviation (so it maintains the output's mean close to 0 and 
#         # it's standard deviation close to 1. This can speed up the training of 
#         # the neural network.
#     model.add(LeakyReLU())

# # Create output layer
# model.add(Dense(1, activation='relu'))


"""Configure the learning process, train the model, save model and it's 
losses, with different learning rates, batch sizes and number of epochs."""
    
# """Configure the learning process of the model with a loss function and an 
# optimizer. The optimizer changes the weights in order to minimize the loss 
# function. In this case we use the Adam optimizer"""
# model = mlp1_put(n_units, n_hidden_layers) 
# model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
    
# """Train the model, given certain hyper-parameters"""
# history = model.fit(puts_X_train, puts_y_train, batch_size = n_batch, 
#                     epochs = n_epochs, validation_split = 0.01, verbose = 1)

# # Save the model's architecture, weights and optimizer's state
# model.save('Saved_models/mlp1_put_1')

# # Save the model's train and validation losses for each epoch.
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_put_1_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_put_1_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")


"""Model using the hyper-parameters in Ke and Yang (2019). LR changes with the 
number of epochs, batch size = 4096, epochs = 30"""
step = tf.Variable(0, trainable = False)
boundaries = [10, 20]
values = [1e-3, 1e-4, 1e-5]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

learning_rate = learning_rate_fn(step)

model = mlp1_put(n_units_paper, n_hidden_layers_paper) 
model.compile(loss='mse', optimizer = keras.optimizers.Adam(
                                        lr = learning_rate))
history = model.fit(puts_X_train, puts_y_train, batch_size = n_batch_paper, 
                    epochs = n_epochs_paper, validation_split = 0.01, 
                    verbose = 1)
model.save('Saved_models/mlp1_put_2')
train_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
numpy__train_loss = np.array(train_loss)
numpy_validation_loss = np.array(validation_loss)
np.savetxt("Saved_models/mlp1_put_2_train_losses.txt", 
            numpy__train_loss, delimiter=",")
np.savetxt("Saved_models/mlp1_put_2_validation_losses.txt", 
            numpy_validation_loss, delimiter=",")


# # LR = 1e-4, batch size = 4096, epochs = n_epochs
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-4))
# history = model.fit(puts_X_train, puts_y_train, batch_size=4096, 
#                     epochs=n_epochs, validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_put_2')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_put_2_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_put_2_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-5, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-5))
# history = model.fit(puts_X_train, puts_y_train, batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_put_3')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_put_3_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_put_3_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # LR = 1e-6, batch size = 4096, epochs = 10
# model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=1e-6))
# history = model.fit(puts_X_train, puts_y_train, 
#                     batch_size=4096, epochs=10, 
#                     validation_split = 0.01, verbose=1)
# model.save('Saved_models/mlp1_put_4')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_put_4_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_put_4_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

# # QUICK TEST
# model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
# history = model.fit(puts_X_train, puts_y_train, batch_size = n_batch, 
#                     epochs = 1, validation_split = 0.01, verbose = 1)
# model.save('Saved_models/mlp1_put_test')
# train_loss = history.history["loss"]
# validation_loss = history.history["val_loss"]
# numpy__train_loss = np.array(train_loss)
# numpy_validation_loss = np.array(validation_loss)
# np.savetxt("Saved_models/mlp1_put_test_train_losses.txt", 
#             numpy__train_loss, delimiter=",")
# np.savetxt("Saved_models/mlp1_put_test_validation_losses.txt", 
#             numpy_validation_loss, delimiter=",")

