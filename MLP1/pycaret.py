# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:20:52 2021

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


from pycaret.utils import version
version()
from os import path
import pandas as pd



# Create DataFrame (DF) for calls
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", 
                                  "Processed data/options_phase3_final.csv"))
df = pd.read_csv(filepath)
# df = df.dropna(axis=0)
df = df.drop(columns=['bid_eod', 'ask_eod', "QuoteDate"])
# df.strike_price = df.strike_price / 1000
calls_df = df[df.OptionType == 'c'].drop(['OptionType'], axis=1)


import pycaret.classification as pc
clf1 = pc.setup(calls_df, target = 'Option_Average_Price', session_id=123, 
             log_experiment=True)

best_model = pc.compare_models()

# mlp = pc.create_model('mlp')

