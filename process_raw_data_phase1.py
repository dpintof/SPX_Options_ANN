"""
Created on Wed Feb 24 07:39:40 2021

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


from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np


print("3 lengthy commands will follow, with respective progress bars:")


# UNDERLYING ASSET
# Create dataframe (df) for the data of the underlying from December 2003 to 
# April 2019
underlying = pd.read_csv("Raw data/Underlying/SPX_December_2003-April_2019.csv")

# Convert dates on Date columns to datetime64
underlying['Date'] = pd.to_datetime(underlying['Date'])

# Sort underlying df by Date column, in ascending order
underlying = underlying.sort_values(by='Date')

# Create new column with the standard deviation of the returns from the past 20 
# days 
underlying['Sigma_20_Days'] = underlying[" Close"].rolling(20).apply(lambda x:
                                                (np.diff(x) / x[:-1]).std())

# Creates new column with the annualized standard deviation of the returns from 
# the past 20 days
underlying['Sigma_20_Days_Annualized'] = underlying['Sigma_20_Days'] * 250**0.5

# Remove unnecessary columns
underlying = underlying.drop([" Open", " High", " Low", "Sigma_20_Days"], 
                             axis = 1)


# TREASURY RATES
# Create df for treasury rates from January 2004 to December 2019
treasury = pd.read_csv("Raw data/Treasury/Treasury_rates_2004-2019.csv", 
                       index_col = "Date")  

# List with the different maturities, in years, of Treasury bonds
treasury_maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

# Change column names of treasury df
treasury_columns = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
treasury.columns = treasury_columns

# Convert data on Date column of treasury df to datetime64
treasury.index = pd.to_datetime(treasury.index)

# Fill the 3-month treasury rate for 2008-12-10, 2008-12-18, 2008-12-23 and 
# 2010-10-11 with the rates for an adjacent day.
treasury.loc[pd.to_datetime("2008-12-10"), 
             0.25] = treasury.loc[pd.to_datetime("2008-12-09"), 0.25]
treasury.loc[pd.to_datetime("2008-12-18"), 
             0.25] = treasury.loc[pd.to_datetime("2008-12-17"), 0.25]
treasury.loc[pd.to_datetime("2008-12-24"), 
             0.25] = treasury.loc[pd.to_datetime("2008-12-23"), 0.25]
treasury.loc[pd.to_datetime("2010-10-11"), 
             0.25] = treasury.loc[pd.to_datetime("2010-10-08"), 0.25]


# OPTIONS
# Set the path to files for options from January 2004
p = Path("Raw data/Options/SPX_20040102_20190430")

# Create a list of the option files
options_files = list(p.glob("UnderlyingOptionsEODQuotes_*.csv"))

# Creates df from all files
# options = pd.concat([pd.read_csv(f) for f in options_files])
options = pd.concat([pd.read_csv(f) for f in tqdm(options_files)])

# Deletes rows for options with a type of option that isn't SPX or SPXW
options = options.loc[options['root'].isin(["SPX", "SPXW"])]

# print(f"Total number of options: {options.shape[0]}")
n_options = options.shape[0] # Total number of options

# Remove unnecessary columns
options = options.drop(['underlying_symbol', 'root', 'open', 'high', 'low', 
                          "close", 'trade_volume', 'bid_size_1545', 'bid_1545',
                          'ask_size_1545', 'ask_1545', 'underlying_bid_1545', 
                          'underlying_ask_1545', 'bid_size_eod', "ask_size_eod",
                          "vwap", "open_interest", "delivery_code"], axis = 1)

# Rename columns quote_date to QuoteDate and option_type to OptionType
options = options.rename(columns = {'quote_date': 'QuoteDate', 
                                      "option_type": "OptionType"})

# Function that returns the number of years between two dates
def years_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days / 365)
    
# CALCULATE THE TIME TO MATURITY (TTM), IN YEARS, FOR EACH OPTION
ttm = []

# for index, row in options.iterrows():
for index, row in tqdm(options.iterrows(), total = n_options):
    d1 = row.expiration
    d2 = row.QuoteDate
    d = years_between(d1, d2)
    ttm.append(d)
    
# Create new column with the TTM
options['Time_to_Maturity'] = ttm

# Calculate the average of the bid and ask prices of the option
option_average_price = []

# for index, row in options.iterrows():
for index, row in tqdm(options.iterrows(), total = n_options):
    bid = row.bid_eod
    ask = row.ask_eod
    average = (ask + bid) / 2
    option_average_price.append(average)
    
# Create new column with the average price
options['Option_Average_Price'] = option_average_price

# Convert data on QuoteDate column of options df to datetime64
options['QuoteDate'] = pd.to_datetime(options['QuoteDate'])


underlying.to_csv("Processed data/underlying.csv", index = False)
treasury.to_csv("Processed data/treasury.csv", index = True)
options.to_csv("Processed data/options_phase1.csv", index = False)
    
