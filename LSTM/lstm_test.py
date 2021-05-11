# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:09:48 2021

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
from sklearn.model_selection import train_test_split, GridSearchCV
from os import path
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


"""Create DFs for options and underlying"""
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
options_df = pd.read_csv(filepath)
options_df = options_df.drop(columns=['Sigma_20_Days_Annualized', 
                                      "Underlying_Price", "bid_eod",
                                      "ask_eod"])
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/underlying.csv"))
underlying_df = pd.read_csv(filepath)

# """Rescaling of the data"""
# underlying_df[" Close"] = MinMaxScaler().fit_transform(underlying_df[" Close"])
# underlying_df[" Close"] = minmax_scale(underlying_df[" Close"])
# options_df["strike"] = minmax_scale(options_df["strike"])
# options_df["Option_Average_Price"] = minmax_scale(options_df["Option_Average_Price"])


# Hyperparameters
n_hidden_layers = 3
n_features = 4 
"""Features: Strike price, time to maturity, risk-free rate and price of the 
underlying."""

n_units = 100 # Number of neurons of the hidden layers in the MLP1 network.
n_units_lstm = 8 # Number of neurons of the LSTM layers
n_batch = 4096
n_epochs_calls = 10
n_epochs_puts = 10
N_TIMESTEPS = 20


# Create array with the prices of the underlying, where the first N_TIMESTEPS 
    # entries are nan
padded = np.insert(underlying_df[" Close"].values, 0, 
                   np.array([np.nan] * N_TIMESTEPS))

# Create Dataframe (df) where the first column is a date and the rest are 
    # prices of the underlying. Each row has N_TIMESTEPS prices of the 
    # underlying, ordered by date, in descending order, from left to right. 
    # From one row to the next each price is replaced by the one observed in 
    # the next date.
rolled = np.column_stack([np.roll(padded, i) for i in range(N_TIMESTEPS)])
rolled = rolled[~np.isnan(rolled).any(axis=1)]
rolled = np.column_stack((underlying_df.Date.values[N_TIMESTEPS - 1:], rolled))
price_history = pd.DataFrame(data=rolled)

cols = price_history.columns.drop(0)
price_history[cols] = price_history[cols].apply(pd.to_numeric, errors='coerce')
    # The last 2 rows are necessary because if the prices of the underlying are
        # not in a numeric data type, Keras' fit function will not work.

"""Add columns of price_history to options_df, according to the date in 
"QuoteDate"""
joined = options_df.join(price_history.set_index(0), on = 'QuoteDate')

# Create dfs for calls and puts
call_df = joined[joined.OptionType == 'c'].drop(['OptionType'], axis=1)
call_df = call_df.drop(columns=['QuoteDate'])
put_df = joined[joined.OptionType == 'p'].drop(['OptionType'], axis=1)
put_df = put_df.drop(columns=['QuoteDate'])


# Split DFs into random train and test arrays, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    call_df.drop(["Option_Average_Price"], axis=1).values, 
    call_df.Option_Average_Price.values, test_size=0.01))
put_X_train, put_X_test, put_y_train, put_y_test = (train_test_split(
    put_df.drop(["Option_Average_Price"], axis=1).values, 
    put_df.Option_Average_Price.values, test_size=0.01))

"""Create lists composed of 2 items: a 3-dimensional array containing 
N_TIMESTEPS columns of prices of the underlying per row and an array with 
n_feature columns with their respective values per row."""
call_X_train = [call_X_train[:, -N_TIMESTEPS:].reshape(call_X_train.shape[0], 
                                N_TIMESTEPS, 1), call_X_train[:, :n_features]]
    # call_X_train[:, -N_TIMESTEPS:] returns an array with the last N_TIMESTEPS
        # columns of call_X_train. Meaning the columns with the prices of the 
        # underlying.
    # call_X_train.shape[0] returns the number of rows of call_X_train
    # call_X_train[:, :n_features] returns an array with the n_features 
        # columns of call_X_train.
call_X_test = [call_X_test[:, -N_TIMESTEPS:].reshape(call_X_test.shape[0], 
                                N_TIMESTEPS, 1), call_X_test[:, :n_features]]
put_X_train = [put_X_train[:, -N_TIMESTEPS:].reshape(put_X_train.shape[0], 
                                N_TIMESTEPS, 1), put_X_train[:, :n_features]]
put_X_test = [put_X_test[:, -N_TIMESTEPS:].reshape(put_X_test.shape[0], 
                                N_TIMESTEPS, 1), put_X_test[:, :n_features]]


# Create model using Keras' functional API
def make_model():

    # Create input layer. The inputs are the closing prices of the underlying for 
        # the past 20 days
    close_history = keras.Input(shape = (N_TIMESTEPS, 1))

    # Create LSTM layers
    lstm = layers.LSTM(units = n_units_lstm, input_shape=(N_TIMESTEPS, 1), 
                    return_sequences=True)(close_history)
    lstm = layers.LSTM(units = n_units_lstm, return_sequences=True)(lstm)
    lstm = layers.LSTM(units = n_units_lstm, return_sequences=True)(lstm)
    lstm = layers.LSTM(units = n_units_lstm, return_sequences=False)(lstm)
    # lstm = layers.Bidirectional(layers.LSTM(units = n_units_lstm, 
    #     input_shape=(N_TIMESTEPS, 1), return_sequences=True))(close_history)
    # lstm = layers.Bidirectional(layers.LSTM(units = n_units_lstm, 
    #                                         return_sequences=True))(lstm)
    # lstm = layers.Bidirectional(layers.LSTM(units = n_units_lstm, 
    #                                         return_sequences=True))(lstm)
    # lstm = layers.Bidirectional(layers.LSTM(units = n_units_lstm, 
    #                                         return_sequences=False))(lstm)

    # Create layer that concatenates the output of the LSTM network (input1) with 
        # the other inputs (input2) necessary to use the MLP1 architecture.
    input1 = lstm
    input2 = keras.Input(shape = (n_features,))
    x = layers.Concatenate()([input1, input2])

    # Function that creates a hidden layer by taking a tensor as input and 
        # applying Batch Normalization and the LeakyReLU activation. The MLP1 
        # hidden layers.
    def hl(tensor):
        dense = layers.Dense(n_units)
        # Dense() creates a densely-connected NN layer, implementing the 
            # following operation: output = activation(dot_product(input, 
            # kernel) + bias) where activation is the element-wise activation 
            # function passed as the activation argument, kernel is a weights 
            # matrix created by the layer, and bias is a bias vector created by
            # the layer (only applicable if use_bias is True, which it is by 
            # default). In this case no activation function was passed so the
            # activation is "linear", meaning: activation(x) = x.
        x = dense(tensor)
        bn = layers.BatchNormalization()(x)
        # Batch normalization scales the output of a layer by subtracting the 
            # batch mean and dividing by the batch standard deviation (so it 
            # maintains the output's mean close to 0 and it's standard 
            # deviation close to 1). Theoretically this can speed up the 
            # training of the neural network.
        lr = layers.LeakyReLU()(bn)
        return lr

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x)

    # Create output layer
    output = layers.Dense(1, activation='relu')(x)    

    # Actually create the model
    return keras.Model(inputs = [close_history, input2], outputs = output)


model = KerasClassifier(build_fn = make_model, epochs = n_epochs_calls)


param_grid = dict(epochs = [10, 20, 30, 40])
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, 
                    cv = 3)
grid_result = grid.fit(call_X_train, call_y_train)

