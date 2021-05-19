# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:42:34 2021

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
from os import path
from tensorflow.keras import layers
from tensorflow import keras
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
import wrangle


# Create DataFrame (DF) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)
frames = [calls_df.iloc[:, :2], calls_df.iloc[:, 3:], calls_df.iloc[:, 2]]
calls_df = pd.concat([frames[0], frames[1]], axis = 1)
calls_df = pd.concat([calls_df, frames[2]], axis = 1)
X = calls_df.iloc[:, 0:5]
y = calls_df.iloc[:, 5]

"""Split call_df into random train and validation subsets, for both inputs(X) 
and output (y)"""
call_X_train, call_y_train, call_X_val, call_y_val = wrangle.array_split(X, y, 
                                                                         .5)

# space = {'choice': hp.choice('num_layers',
#                     [ {'layers':'two', },
#                     {'layers':'three',
#                     'units3': hp.uniform('units3', 64,1024), 
#                     'dropout3': hp.uniform('dropout3', .25,.75)}
#                     ]),

#             'units1': hp.uniform('units1', 64,1024),
#             'units2': hp.uniform('units2', 64,1024),

#             'dropout1': hp.uniform('dropout1', .25,.75),
#             'dropout2': hp.uniform('dropout2',  .25,.75),

#             'batch_size' : hp.uniform('batch_size', 28,128),

#             'nb_epochs' :  100,
#             'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
#             'activation': 'relu'
#         }

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,1024), 
                    'dropout3': hp.uniform('dropout3', .25,.75)}
                    ]),

            'units1': hp.uniform('units1', 64,1024),
            'units2': hp.uniform('units2', 64,1024),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }


"""Function that creates a hidden layer by taking a tensor as input and 
applying Batch Normalization and the LeakyReLU activation."""
def hl(tensor, n_units):
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
def make_model(params):
    
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU()(inputs)

    # Create hidden layers
    for _ in range(params["n_hidden_layers"]):
        x = hl(x, params["n_units"])
        
    # Create output layer
    outputs = layers.Dense(1, activation='relu')(x)

    # Actually create the model
    model = keras.Model(inputs = inputs, outputs = outputs)
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3))
    model.compile(loss = 'mse', 
       optimizer = keras.optimizers.Adam(lr = params["learning_rate"]))
    # model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(), 
    #               metrics=["accuracy"]) # added metrics to see less warnings
    
    # Train the model
    model.fit(X, y, batch_size = params["n_batch"], epochs = params["n_epochs"])
    
    pred_auc = model.predict_proba(call_X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(call_y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


# def f_nn(params):   
#     from keras.models import Sequential
#     from keras.layers.core import Dense, Dropout, Activation
#     from keras.optimizers import Adadelta, Adam, rmsprop

#     print ('Params testing: ', params)
#     model = Sequential()
#     model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
#     model.add(Activation(params['activation']))
#     model.add(Dropout(params['dropout1']))

#     model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
#     model.add(Activation(params['activation']))
#     model.add(Dropout(params['dropout2']))

#     if params['choice']['layers']== 'three':
#         model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
#         model.add(Activation(params['activation']))
#         model.add(Dropout(params['choice']['dropout3']))    

#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

#     model.fit(X, y, nb_epoch=params['nb_epochs'], 
#               batch_size=params['batch_size'], verbose = 0)

#     pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
#     acc = roc_auc_score(y_val, pred_auc)
#     print('AUC:', acc)
#     sys.stdout.flush() 
#     return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(make_model, space, algo=tpe.suggest, max_evals=50, trials=trials)
print ('best: ')
print (best)