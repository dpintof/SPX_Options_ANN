# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:38:26 2021

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


from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from os import path
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from multiprocessing import cpu_count


# print(f"The CPUs has {cpu_count()} cores.")


# Hyperparameters
n_hidden_layers = 3
n_units = 400 # Number of neurons of the hidden layers.
n_batch = 1024 # Number of observations used per gradient update.
n_epochs = 40


# Create DataFrame (DF) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01)
    )


"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor):
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
def make_model(n_hidden_layers = n_hidden_layers, n_units = n_units, 
               n_batch = n_batch, n_epochs = n_epochs, learning_rate = 1e-3):
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x)
        
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3))
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(), 
                  metrics=["accuracy"]) # added metrics to see less warnings
    return model


model = KerasClassifier(build_fn = make_model, n_epochs = n_epochs)


# param_grid = dict(n_hidden_layers = np.arange(1, 10, 1), 
#                   n_units = np.arange(100, 1000, 100), 
#                   n_batch = np.arange(1024, 10240, 1024), 
#                   n_epochs = [10, 20, 30, 40], 
#                   learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
# grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3,
                    # n_jobs = 1, verbose = 1)
                    
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
# param_dist = {"n_hidden_layers": np.random.randint(low = 1, high = 11), 
#               "n_units": np.random.randint(low = 1, high = 1001),
#               "n_batch": np.random.randint(low = 1024, high = 10241),
#               "n_epochs": np.random.randint(low = 10, high = 41), 
#               "learning_rate": np.random.uniform(low = 1e-6, high = 1e-1)}
param_dist = {"n_hidden_layers": sp_randInt(1, 11), 
              "n_units": sp_randInt(1, 1001),
              "n_batch": sp_randInt(1024, 10241),
              "n_epochs": sp_randInt(10, 41), 
              "learning_rate": sp_randFloat(1e-6, 1e-1)}
n_iter_search = 30
grid = RandomizedSearchCV(estimator = model, 
                          param_distributions = param_dist, 
                          n_iter = n_iter_search, verbose = 3)

grid_result = grid.fit(call_X_train, call_y_train)
print(grid.best_score_)
print(grid.best_params_)

