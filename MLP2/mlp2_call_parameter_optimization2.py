# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:07:49 2021

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
from skopt import BayesSearchCV  # skopt = scikit-optimize
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer
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


model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = make_model)


def fit_with(n_hidden_layers, n_neurons, learning_rate, batch_size, n_epochs):

    # # Create the model
    # model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = make_model)

    # Train the model with the train dataset.
    model.fit(x=call_X_train, y=call_y_train, epochs=n_epochs, 
              batch_size=batch_size)

    # Evaluate the model with the test dataset.
    score = model.evaluate(call_X_test, steps=10,)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the loss.
    return score[0]


model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = fit_with)


# #
# def model_compile_fit(n_hidden_layers, n_neurons, batch_size, n_epochs, 
#                       learning_rate):
#     model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = make_model)
    
#     model.compile(loss='mse', optimizer = keras.optimizers.Adam(
#                                                 learning_rate=learning_rate))
    
#     # history = model.fit(call_X_train, call_y_train, 
#     #                     batch_size=batch_size, epochs=n_epochs, 
#     #                     validation_split = 0.01, verbose=1)
    
#     # return history
#     return model

params1 = {"n_neurons": Integer(200, 900, prior="uniform"),
        "n_hidden_layers": Integer(2, 6, prior="uniform"),
        "learning_rate": Real(1e-5, 1, prior="uniform"),
          }
# params2 = {"batch_size": Integer(1024, 10240, prior="uniform"), 
#            "n_epochs": Integer(10, 45, prior="uniform"),
#           }

params3 = {"n_neurons": Integer(200, 900, prior="uniform"),
           "n_hidden_layers": Integer(2, 6, prior="uniform"),
           "learning_rate": Real(1e-5, 1, prior="uniform"),
           "batch_size": Integer(1024, 10240, prior="uniform"), 
           "n_epochs": Integer(10, 45, prior="uniform"),
          } 

# Bayesian optimizer
# opt = BayesSearchCV(model, search_spaces=params1, n_iter=32, verbose=1,)
opt = BayesSearchCV(model, search_spaces=params3, n_iter=32, verbose=1,)


# Executes bayesian optimization
_ = opt.fit(call_X_train, call_y_train,)

# Model can be saved, used for predictions or scoring
print(opt.score(call_X_test, call_y_test))

