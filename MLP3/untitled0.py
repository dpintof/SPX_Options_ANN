#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:19:09 2021

@author: Diogo Pinto
"""

import numpy as np
import tensorflow as tf
import pandas as pd


np.random.seed(0)
inp = np.random.rand(2, 5)
inp_tf = tf.convert_to_tensor(inp, np.float32)

df = pd.DataFrame(inp, columns = ['a','b','c', 'd', "e"])
df_tf = tf.convert_to_tensor(df, np.float32)