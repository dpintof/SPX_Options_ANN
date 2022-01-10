# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 07:39:35 2021

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
from tqdm import tqdm


# LOAD DATA
underlying = pd.read_csv("Processed data/underlying.csv")
treasury = pd.read_csv("Processed data/treasury.csv")
options = pd.read_csv("Processed data/options_phase1.csv")

# Convert dates to datetime64
options['QuoteDate'] = pd.to_datetime(options['QuoteDate'])
treasury["Date"] = pd.to_datetime(treasury["Date"])


# GET A 3-MONTH (0,25 YEARS) RISK-FREE RATE THAT MATCH EACH OPTION'S QUOTEDATE
# Create list with the 3-month Treasury rates that match each option's 
# QuoteDate
n_options = options.shape[0] # Total number of options

rf_rate = []
QuoteDate_df = options[['QuoteDate']]
QuoteDate_df = (QuoteDate_df.assign(in_treasury = 
                                QuoteDate_df.QuoteDate.isin(treasury.Date)))

for index, row in tqdm(QuoteDate_df.iterrows(), total = n_options):
# for index, row in tqdm(options.iterrows()):
    if row.in_treasury == True:
        (rf_rate.append(float(treasury["0.25"].loc[(treasury["Date"] 
                                                    == row.QuoteDate)])))
    else:
        # This situation happens if the treasury DF doesn't have a rate for a 
        # date in which options exist. -4 is just an arbitrary negative number
        rf_rate.append(-4)

# Add rf_rate as a column in the options df and drop unnecessary columns
options["RF_Rate"] = rf_rate

# Remove options with rf rate = -4
indices = options[options['RF_Rate'] == -4].index
options.drop(indices ,inplace = True)


# CREATE CSV FILE FROM THE OPTIONS DF
options.to_csv('Processed data/options_phase2.csv', index = False)

