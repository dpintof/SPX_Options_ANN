# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:39:16 2021

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
import random


# Create DataFrame (DF) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)

print(calls_df.info())

# Visualize data
# calls_df.iloc[:47000, 0].plot()

# random_integer = random.randint(1, calls_df.shape[0] - 100000)
# calls_df.iloc[round(random_integer/2):round((random_integer + 100000)/2), 
#                                                               5].plot()

