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


"""Prevents Tensorflow warnings about CPU instrucions"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Save the output of the script to a file
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("MLP1_call_optimization_log.txt", "w")


from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from os import path
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from multiprocessing import cpu_count
from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers


# print(f"The CPUs has {cpu_count()} cores.")


# Hyper-parameters
n_hidden_layers = 3
n_units = 400 # Number of neurons of the hidden layers.
n_batch = 4096 # Number of observations used per gradient update.
n_epochs = 30


# Create DataFrame (DF) for calls
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


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    calls_df.drop(["Option_Average_Price"], axis = 1), 
    calls_df.Option_Average_Price, test_size = 0.01)
    )

# """Alternative data for tests"""
# call_X_train, call_y_train = make_classification(n_samples=1000, n_classes=2,
#                             n_informative=4, weights=[0.7, 0.3],
#                             random_state=0)


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
# def make_model(n_hidden_layers, n_units, learning_rate):
# def make_model(num_layers, num_neurons, learning_rate):
def make_model(num_layers, num_neurons):  
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(num_layers):
        x = hl(x)
        
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3))
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
    
    # Added accuracy metric to avoid error in .fit method later. Just ignore it.
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(), 
                    metrics=["accuracy"])
        
    return model


# def make_model(num_neurons=64,num_layers=4,input_dim=20,
#                    output_dim=2,learning_rate=1.0e-05,act='relu',
#                    dropout=0.3):
#     model = Sequential()

#     model.add(Dense(num_neurons,activation='relu',input_dim=input_dim))

#     for i in range(1,num_layers):
#         model.add(Dense(num_neurons,activation=act))

#     model.add(Dropout(dropout))

#     model.add(Dense(output_dim,activation='softmax'))

#     adam = optimizers.Adam(lr=learning_rate)

#     model.compile(adam,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy']
#                  )
#     return model

model = KerasClassifier(build_fn = make_model)
# model = KerasClassifier(build_fn = make_model, epochs = n_epochs) 
"""Need to have the epochs parameter or it returns an error. In the 
optimization process epochs are variable anyway"""
# model = KerasClassifier(build_fn = make_model, epochs = 1,
#                     batch_size = np.random.randint(low = 1, high = 10241))
# 

# batch_size = [16,32,64]
batch_size = np.arange(1024, 10241, 1024).tolist()
# epochs = [1]
epochs = np.arange(10, 46, 5).tolist()
# num_neurons = [6,1,2]
num_neurons = np.arange(300, 901, 100).tolist()
# n_hidden_layers = [1,2]
n_hidden_layers = np.arange(2, 7, 1).tolist()
# learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# dropout = [0.1,0.3,0.5]
param_grid = dict(batch_size=batch_size,
                   epochs=epochs,
                      num_neurons=num_neurons,
                      num_layers=n_hidden_layers,
                      # learning_rate=learning_rate
                      )

# from scipy.stats import uniform as sp_randFloat
# from scipy.stats import randint as sp_randInt
# # param_dist = {"n_hidden_layers": np.random.randint(low = 1, high = 11), 
# #               "n_units": np.random.randint(low = 1, high = 1001),
# #               "n_batch": np.random.randint(low = 1024, high = 10241),
# #               "n_epochs": np.random.randint(low = 10, high = 41), 
# #               "learning_rate": np.random.uniform(low = 1e-6, high = 1e-1)}
# param_grid = {"num_layers": sp_randInt(1, 5), 
#               "num_neurons": sp_randInt(1, 11),
#                 # "n_batch": sp_randInt(1, 10241),
#                 # "n_epochs": sp_randInt(10, 41), 
#                 "learning_rate": sp_randFloat(1e-6, 1e-1)
#               }


n_iter_search = 10

grid = RandomizedSearchCV(estimator = model, 
                          param_distributions = param_grid, 
                          n_iter = n_iter_search, verbose = 3, cv = 2)

# grid = RandomizedSearchCV(estimator = model, 
#                           param_distributions = param_grid, 
#                           n_iter = n_iter_search, verbose = 2, cv = 2, 
#                           scoring="neg_mean_squared_error")

# from sklearn.metrics import mean_squared_error, make_scorer
# mse_scorer = make_scorer(mean_squared_error)
# grid = RandomizedSearchCV(estimator = model, 
#                           param_distributions = param_grid, 
#                           n_iter = n_iter_search, verbose = 2, cv = 2, 
#                           scoring=mse_scorer)

"""The cv parameter refers to k-fold cross validation, as explained in Liu et 
al. (2019), page 9"""

optimization_results = grid.fit(call_X_train, call_y_train)
# print(1, grid.best_score_)
# print("Best score: ", grid.best_score_)
# print(2, grid.best_params_)
# print("Best set of hyper-parameters: ", grid.best_params_)
# print(3, grid.cv_results_)

# cv_results = grid.cv_results_
# for mean_score in zip(cv_results["mean_test_score"]):
#                        print(-mean_score)


# Close log file
# sys.stdout.close()

