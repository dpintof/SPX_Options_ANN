# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:30:48 2021

@author: Diogo Pinto
"""

# Clear the console and remove all variables present on the namespace.
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


# Imports
from os import path
import pandas as pd
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split


# Create DataFrame (df) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
df = df.drop(columns=['Option_Average_Price', "QuoteDate"])
# df = df.drop(columns=["QuoteDate"])
call_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(
    call_df.drop(["bid_eod", "ask_eod"], axis = 1), call_df[["bid_eod", 
                                        "ask_eod"]], test_size = 0.01)


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
def make_model(n_hidden_layers, n_neurons, learning_rate):  
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x, n_neurons)
        
    # Create output layer
    outputs = layers.Dense(2, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(
                                                learning_rate=learning_rate), 
                  metrics=["accuracy"]
                  )
    # Added accuracy metric to avoid error in the fit method.
        
    return model


def fit_with(n_hidden_layers, n_neurons, learning_rate, batch_size, n_epochs, 
             verbose):

    # Create the model using a specified hyperparameters.
    # model = get_model(input_shape, dropout2_rate)
    model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = make_model)

    # Train the model for a specified number of epochs.
    model.compile(loss='mse', 
                  optimizer = keras.optimizers.Adam(learning_rate=learning_rate))


    # Train the model with the train dataset.
    model.fit(call_X_train, call_y_train, 
                    batch_size=batch_size, epochs=n_epochs, 
                    validation_split = 0.01, verbose=1)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(call_X_test, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the loss.
    return -score[0] # Added "-" because we will maximize that amount

from functools import partial

verbose = 1
input_shape = call_X_train.shape[1]
fit_with_partial = partial(fit_with, input_shape, verbose=1)

# Bayesian Optimization explained: https://github.com/fmfn/BayesianOptimization
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {"n_hidden_layers": (2, 6),
           "n_neurons": (200, 900),
           "learning_rate": (1e-5, 1),
           "batch_size": (1024, 10240), 
           "n_epochs": (10, 45),
           }

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=10, n_iter=10,)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)

