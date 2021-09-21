# Clear the console and remove all variables present on the namespace
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, losses
import numpy as np


# Hyperparameters
n_hidden_layers = 2 # Number of hidden layers.
n_units = 128 # Number of neurons of the hidden layers.
n_batch = 64 # Number of observations used per gradient update.
n_epochs = 30


# Sample data
x_train = {'strike':  [200, 2925], 'Time to Maturity': [0.312329, 0.0356164], 
        "RF Rate": [0.08, 2.97], 
        "Sigma 20 Days Annualized": [0.123251, 0.0837898], 
        "Underlying Price": [1494.82, 2840.69]
        }

call_X_train = pd.DataFrame(x_train, columns = ['strike', "Time to Maturity", 
                                                  "RF Rate", 
                                                  "Sigma 20 Days Annualized", 
                                                  "Underlying Price"]
                            )

x_test = {'strike':  [200], 'Time to Maturity': [0.0356164], 
        "RF Rate": [2.97], 
        "Sigma 20 Days Annualized": [0.0837898], 
        "Underlying Price": [2840.69]
        }

call_X_test = pd.DataFrame(x_test, columns = ['strike', "Time to Maturity", 
                                                  "RF Rate", 
                                                  "Sigma 20 Days Annualized", 
                                                  "Underlying Price"]
                           )

y_train = np.array([1285.25, 0.8])
call_y_train = pd.Series(y_train)

y_test = np.array([0.8])
call_y_test = pd.Series(y_test)


# Creates hidden layers
def hl(tensor, n_units):
    hl_output = layers.Dense(n_units, 
                             activation = layers.LeakyReLU(alpha = 1))(tensor)
    # alpha = 1 makes the function LeakyReLU C^inf
    return hl_output

# Create model using Keras' Functional API
def mlp3_call(n_hidden_layers, n_units):
    # Create input layer
    inputs = keras.Input(shape = (call_X_train.shape[1],))
    x = layers.LeakyReLU(alpha = 1)(inputs)

    # Create hidden layers
    for _ in range(n_hidden_layers):
        x = hl(x, n_units)

    # Create output layer
    outputs = layers.Dense(1, activation = keras.activations.softplus)(x)

    # Actually create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# Custom loss function
def constrained_mse(y_true, y_pred):
    
    mse = losses.mse(y_true, y_pred)
    
    # x = tf.constant(call_X_train, dtype=tf.float32)
    x = tf.convert_to_tensor(call_X_train, np.float32)
    # x = tf.convert_to_tensor(call_X_train.iloc[:,0:2], np.float32)
    # with tf.GradientTape() as tape:
    #     tape.watch(x)
    #     y = model(x)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            y = model(x)
        
            # g = tape.gradient(y, x)[:,0]
            # g = tape.gradient(y, x)

    # jacobian = tape.jacobian(y, x)
    # print(jacobian)
    
    grad_y = tape2.gradient(y, x)
    dy_dstrike = grad_y[0, 0]
    dy_dttm = grad_y[0, 1]
    
    d2y_dstrike2 = tape.gradient(dy_dstrike, x[:,0])
    
    # grad_y = tape2.gradient(y, x)
    # j = tape.jacobian(grad_y, x)
    # d2y_dstrike2 = j[0, 0, :, 0, 0]
    
    loss = mse + dy_dstrike + dy_dttm
    # loss = mse
    # loss = mse + g

    return loss
    

model = mlp3_call(n_hidden_layers, n_units) 
model.compile(loss = constrained_mse, optimizer = keras.optimizers.Adam(),)
history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
                    epochs = n_epochs, validation_split = 0.01, verbose = 1)

