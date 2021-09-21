# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:19:32 2021

@author: Diogo
"""

# Clear the console and remove all variables present on the namespace. This is 
# useful to prevent Python from consuming more RAM each time I run the code.
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import tensorflow as tf
import numpy as np
from tensorflow import keras


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

# Penalization function used in both the Custom loss funcion and the metric for 
# arbitrage-free prices
def pen(x, lamb, m):
    return tf.where(x < 0, 0.0, lamb * x**m)

def constrained_mse(y_true, y_pred):
    
    # Parameters of penalization function
    lamb = 10
    m = 4
    
    # Custom loss function
    squared_difference = tf.square(y_true - y_pred)
    return (tf.reduce_mean(squared_difference, axis=-1)

    # return (keras.backend.mean(keras.backend.square(y_pred - y_true)) # MSE
            
            + pen(-(model.input[:,0])**2 * tf.gradients(tf.gradients(y_pred, 
                    model.input), model.input)[0][:, 0], lamb, m) # constraint 1
            
            + pen(-model.input[:,1] * tf.gradients(y_pred, 
                    model.input)[0][:, 1], lamb, m) # constraint 2

            + pen(model.input[:,0] * tf.gradients(y_pred, 
                    model.input)[0][:, 0], lamb, m)) # constraint 3


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)])

model.compile(optimizer='adam', loss=my_loss_fn)

x = np.random.rand(1000)
y = x**2

history = model.fit(x, y, epochs=10)