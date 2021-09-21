#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:17:47 2021

@author: Diogo Pinto
"""


import tensorflow as tf


with tf.GradientTape() as g:
  x = tf.constant([1.0, 2.0])
  g.watch(x)
  y = x * x
jacobian = g.jacobian(y, x)

# print(jacobian[:,1])
print(jacobian)

