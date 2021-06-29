#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:50:03 2021

@author: Diogo
"""

import numpy as np
import matplotlib.pyplot as plt


"""IMPORTANT: DON'T RELY ON THE PLOTS SHOWN IN PYTHON. CHECK THE OUTPUT FILES 
INSTEAD."""


def loss_plot(model, option_type):
    if model == "mlp1":
        if option_type == "call":
            train = np.loadtxt("MLP1/Saved_models/mlp1_call_2_train_losses.txt")
            val = np.loadtxt("MLP1/Saved_models/mlp1_call_2_validation_losses.txt")
        else:
            train = np.loadtxt("MLP1/Saved_models/mlp1_put_2_train_losses.txt")
            val = np.loadtxt("MLP1/Saved_models/mlp1_put_2_validation_losses.txt")
    elif model == "mlp2":
        if option_type == "call":
            train = np.loadtxt("MLP2/Saved_models/mlp2_call_1_train_losses.txt")
            val = np.loadtxt("MLP2/Saved_models/mlp2_call_1_validation_losses.txt")
        else:
            train = np.loadtxt("MLP2/Saved_models/mlp2_put_1_train_losses.txt")
            val = np.loadtxt("MLP2/Saved_models/mlp2_put_1_validation_losses.txt")
    else:
        if option_type == "call":
            train = np.loadtxt("LSTM/Saved_models/lstm_call_3_train_losses.txt")
            val = np.loadtxt("LSTM/Saved_models/lstm_call_3_validation_losses.txt")
        else:
            train = np.loadtxt("LSTM/Saved_models/lstm_put_4_train_losses.txt")
            val = np.loadtxt("LSTM/Saved_models/lstm_put_4_validation_losses.txt")
    
    plt.plot(range(1, train.shape[0]+1), np.log(train))
    plt.plot(range(1, train.shape[0]+1), np.log(val))
    plt.xlabel('Epoch')
    plt.ylabel('log(MSE)')
    plt.legend(['Training Loss', 'Validation Loss'])


loss_plot("mlp1", "call")
plt.title('MLP1 Call Loss')
plt.savefig('MLP1/Saved_models/mlp1_call_losses_plot.png')
plt.cla()

loss_plot("mlp1", "put")
plt.title('MLP1 Put Loss')
plt.savefig('MLP1/Saved_models/mlp1_put_losses_plot.png')
plt.cla()

# loss_plot("mlp2", "call")
# plt.title('MLP2 Call Loss')
# plt.savefig('MLP2/Saved_models/mlp2_call_losses_plot.png')
# plt.cla()

# loss_plot("mlp2", "put")
# plt.title('MLP2 Put Loss')
# plt.savefig('MLP2/Saved_models/mlp2_put_losses_plot.png')
# plt.cla()

# loss_plot("lstm", "call")
# plt.title('LSTM Call Loss')
# plt.savefig('LSTM/Saved_models/lstm_call_losses_plot_3.png')
# plt.cla()

# loss_plot("lstm", "put")
# plt.title('LSTM Put Loss')
# plt.savefig('LSTM/Saved_models/lstm_put_losses_plot_4.png')
# plt.cla()

