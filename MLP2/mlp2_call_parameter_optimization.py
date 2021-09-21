# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 07:05:35 2021

@author: Diogo
"""

# Clear the console and remove all variables present on the namespace.
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


"""Prevents Tensorflow warnings about CPU instructions"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Save the output of the script to a file
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("MLP2_call_optimization_log.txt", "w")


from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from os import path
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# Create DataFrame (DF) for puts
basepath = path.dirname(__file__)
# filepath = path.abspath(path.join(basepath, "..", 
#                                   "Processed data/options_phase3_final.csv"))
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_free_dataset.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_Y_train, call_Y_test = train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01)


"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor, n_neurons):
    dense = layers.Dense(n_neurons)
    """Dense() creates a densely-connected NN layer, implementing the following 
    operation: output = activation(dot_product(input, kernel) + bias) where 
    activation is the element-wise activation function passed as the activation
    argument, kernel is a weights matrix created by the layer, and bias is a 
    bias vector created by the layer (only applicable if use_bias is True, 
    which it is by default). In this case no activation function was passed so 
    there is "linear" activation: a(x) = x."""
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


# Create MLP2 model using Keras' functional API
def make_model(n_layers, n_neurons):  
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(n_layers):
        x = hl(x, n_neurons)
        
    # Create output layer
    outputs = layers.Dense(2, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(), 
                    metrics=["accuracy"])
    # Added accuracy metric to avoid error in fit method later.
        
    return model


model = KerasClassifier(build_fn = make_model)


# batch_size = [16,32,64]
batch_size = np.arange(1024, 10241, 1024).tolist()
# epochs = [1]
n_epochs = np.arange(10, 46, 5).tolist()
# n_neurons = [6,1,2]
n_neurons = np.arange(200, 901, 100).tolist()
# n_hidden_layers = [1,2]
n_hidden_layers = np.arange(2, 7, 1).tolist()
# learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# dropout = [0.1,0.3,0.5]
param_grid = dict(batch_size = batch_size, epochs = n_epochs, 
                  n_neurons = n_neurons, n_layers = n_hidden_layers, 
                  # learning_rate=learning_rate
                  )


n_iter_search = 10

grid = RandomizedSearchCV(estimator = model, 
                          param_distributions = param_grid, 
                          n_iter = n_iter_search, verbose = 3, cv = 2)
"""The cv parameter refers to k-fold cross validation, as explained in Liu et 
al. (2019), page 9"""

optimization_results = grid.fit(call_X_train, call_Y_train)

