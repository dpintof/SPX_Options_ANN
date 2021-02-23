#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:26:20 2021

@author: Diogo
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


def loss_plot(name):
    train = np.loadtxt(f'loss-{name}.txt')
    val = np.loadtxt(f'val-loss-{name}.txt')
    plt.plot(range(1, train.shape[0]+1), np.log(train))
    plt.plot(range(1, train.shape[0]+1), np.log(val))
    plt.xlabel('Epoch')
    plt.ylabel('log(MSE)')
    plt.legend(['Training Loss', 'Validation Loss'])


loss_plot('mlp1-call')
plt.title('MLP1 Call Loss')
plt.savefig('mlp1-call-plot.png')

loss_plot('mlp1-put')
plt.title('MLP1 Put Loss')
plt.savefig('mlp1-put-plot.png')

loss_plot('mlp2-call')
plt.title('MLP2 Call Loss')
plt.savefig('mlp2-call-plot.png')

loss_plot('mlp2-put')
plt.title('MLP2 Put Loss')
plt.savefig('mlp2-put-plot.png')

