# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 06:31:08 2021

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
from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers

# print(f"The CPUs has {cpu_count()} cores.")


# Hyper-parameters
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
puts_df = df[df.OptionType == 'p'].drop(['OptionType'], axis=1)


"""Split call_df into random train and test subsets, for inputs (X) and 
output (y)"""
put_X_train, put_X_test, put_y_train, put_y_test = (train_test_split(
    puts_df.drop(["Option_Average_Price"], axis = 1), 
    puts_df.Option_Average_Price, test_size = 0.01)
    )


"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor):
    dense = layers.Dense(n_units)
    """Dense() creates a densely-connected NN layer, implementing the following 
    operation: output = activation(dot_product(input, kernel) + bias) where 
    activation is the element-wise activation function passed as the activation
    argument, kernel is a weights matrix created by the layer, and bias is a 
    bias vector created by the layer (only applicable if use_bias is True, 
    which it is by default). In this case no activation function was passed so 
    there is "linear" activation: a(x) = x."""
    
    x = dense(tensor)
    bn = layers.BatchNormalization()(x)
    """Batch normalization scales the output of a layer by subtracting the 
    batch mean and dividing by the batch standard deviation (so the output's 
    mean will be close to 0 and it's standard deviation close to 1). 
    Theoretically this can speed up the training of the neural network."""
    
    lr = layers.LeakyReLU()(bn)
    return lr


# Create MLP1 model using Keras' functional API
# def make_model(n_hidden_layers, n_units, learning_rate):
def make_model(num_layers, num_neurons, learning_rate):  
    
    # Create input layer
    inputs = keras.Input(shape = (put_X_train.shape[1],))
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
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(
                    learning_rate = learning_rate), metrics=["accuracy"]) 
                    # added metrics to see less warnings
    return model


model = KerasClassifier(build_fn = make_model,
                    epochs = 1)
                    # batch_size = np.random.randint(low = 1, high = 10241))


# batch_size = [16,32,64]
batch_size = np.arange(1024, 10240, 1024)
# epochs = [2,3] 
epochs = np.arange(5, 40, 5)
# num_neurons = [6,1,2]
num_neurons = np.arange(100, 1000, 100)
# n_hidden_layers = [1,2]
n_hidden_layers = np.arange(1, 10, 1)
# learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 

# dropout = [0.1,0.3,0.5]
param_grid = dict(batch_size=batch_size,
                   epochs=epochs,
                      num_neurons=num_neurons,
                      num_layers=n_hidden_layers)
                      # learning_rate=learning_rate)

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


n_iter_search = 5

grid = RandomizedSearchCV(estimator = model, 
                          param_distributions = param_grid, 
                          n_iter = n_iter_search, verbose = 3) # Check other verbose numbers
"""The cv parameter refers to k-fold cross validation, as explained in Liu et 
al. (2019), page 9"""

grid_result = grid.fit(put_X_train, put_y_train)
print(grid.best_score_)
print(grid.best_params_)
