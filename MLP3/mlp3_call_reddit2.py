# Clear the console and remove all variables present on the namespace
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers, losses
import numpy as np


# Hyperparameters
n_hidden_layers = 2 # Number of hidden layers.
n_units = 128 # Number of neurons of the hidden layers.
n_batch = 64 # Number of observations used per gradient update.
n_epochs = 30


# Create DataFrame (df) with random floats
np.random.seed(30)
call_df = pd.DataFrame(np.random.rand(6000000, 6) * 100, 
                       columns=['Strike', "Time to Maturity", 
                                "Option_Average_Price", "RF Rate", 
                                "Sigma 20 Days Annualized", 
                                "Underlying Price"])
call_df = call_df.iloc[:10,:]

# Split call_df into random train and test subsets, for inputs (X) and output (y)
call_X_train, call_X_test, call_y_train, call_y_test = (train_test_split(
    call_df.drop(["Option_Average_Price"], axis = 1), 
    call_df.Option_Average_Price, test_size = 0.01))

array = call_X_train.values

tensor = tf.convert_to_tensor(array)


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


# Hidden layer generation function
def hl(tensor, n_units):
    hl_output = layers.Dense(n_units, 
                             activation = layers.LeakyReLU(alpha=1))(tensor)
    return hl_output


# Custom loss function that is a MSE function plus three soft constraints
def constrained_mse(y_true, y_pred):
    
    mse = losses.mse(y_true, y_pred)
    
    array = call_X_train.values
    # model.run_eagerly = True
    x = tf.convert_to_tensor(array, dtype="float32")
    print(x)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            y = model(x)
        
    grad_y = tape2.gradient(y, x)
    dy_dstrike = grad_y[0, 0]
    dy_dttm = grad_y[0, 1]
    
    grad_y2 = tape.gradient(y, x)
    d2y_dstrike2 = grad_y2[0, 0]
    
    loss = mse + dy_dstrike + dy_dttm + d2y_dstrike2

    return loss


model = mlp3_call(n_hidden_layers, n_units) 
model.compile(loss = constrained_mse, optimizer = keras.optimizers.Adam(), 
              run_eagerly=True)
history = model.fit(call_X_train, call_y_train, batch_size = n_batch, 
                    epochs = n_epochs, validation_split = 0.01, verbose = 1)