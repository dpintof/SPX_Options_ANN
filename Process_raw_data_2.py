#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 07:39:40 2021

@author: Diogo
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np


# Underlying asset
# Create dataframe (df) for the data of the underlying from August to September 2018
underlying = pd.read_csv("Raw data/Underlying/SPX_December_2003-April_2019.csv")

# Convert dates on Date columns to datetime64
underlying['Date'] = pd.to_datetime(underlying['Date'])

# Sort underlying df by Date column, in ascending order
underlying = underlying.sort_values(by='Date')

# Creates new column with the standard deviation of the returns from the past 20 days
underlying['Sigma_20_Days'] = underlying[" Close"].rolling(20).apply(lambda x:
                                                (np.diff(x) / x[:-1]).std())

# Remove unnecessary columns
underlying = underlying.drop([" Open", " High", " Low"], axis = 1)

# Create csv file from the underlying df
underlying.to_csv('underlying_df.csv', index=False)


# Treasury rates
# Create df for treasury rates from January 2004 to December 2019
treasury = pd.read_csv("Raw data/Treasury/Treasury_rates_2004-2019.csv")  

# List with the different maturities, in years, of Treasury bonds
treasury_maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30] 

# Change column names of treasury df
treasury_columns = ["Date", 1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30] 
treasury.columns = treasury_columns

# Convert data on Date column of treasury df to datetime64
treasury['Date'] = pd.to_datetime(treasury['Date'])


# Options
# Set the path to files for options from August 2019
p = Path("Raw data/Options/SPX_20140102_20190430")

# Create a list of the option files from August 2019
options_files = list(p.glob("UnderlyingOptionsEODQuotes_*.csv"))

# Creates df from all files
options = pd.concat([pd.read_csv(f) for f in options_files]) 
# TESTING WITH A SMALL SAMPLE
# options = pd.read_csv("Raw data/Options/SPX_20140102_20190430/"
#                        "UnderlyingOptionsEODQuotes_2004-01-02.csv")
# options = options.iloc[8:]

# Deletes rows for options with a type of option that isn't SPX or SPXW
options = options.loc[options['root'].isin(["SPX", "SPXW"])]


# Remove unnecessary columns
options = options.drop(['underlying_symbol', 'root', 'open', 'high', 'low', 
                          "close", 'trade_volume', 'bid_size_1545', 'bid_1545',
                          'ask_size_1545', 'ask_1545', 'underlying_bid_1545', 
                          'underlying_ask_1545', 'bid_size_eod', "ask_size_eod",
                          "vwap", "open_interest", "delivery_code"], axis = 1)

# Rename columns DataDate to QuoteDate and Type to OptionType
options = options.rename(columns = {'quote_date': 'QuoteDate', 
                                      "option_type": "OptionType"})

# Create function that returns the number of years between two dates
def years_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days / 365)
    
# Calculate the time to maturity (TTM), in years, for each option
ttm = []

for index, row in options.iterrows():
    d1 = row.expiration
    d2 = row.QuoteDate
    d = years_between(d1, d2)
    ttm.append(d)
    
# Create new column with the TTM
options['Time_to_Maturity'] = ttm

# Calculate the average of the bid and ask prices of the option
option_average_price = []

for index, row in options.iterrows():
    bid = row.bid_eod
    ask = row.ask_eod
    average = (ask + bid) / 2
    option_average_price.append(average)
    
# Create new column with the average price
options['Option_Average_Price'] = option_average_price

# Convert data on quote_date column of options df to datetime64
options['QuoteDate'] = pd.to_datetime(options['QuoteDate'])

# Subtract the TTM of each option with the different maturities
differences = pd.DataFrame(columns = treasury_maturities)
treasury_maturities1 = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
treasury_maturities2 = [1/12]
treasury_maturities3 = [6/12, 1, 2, 3, 5, 7, 10, 20, 30]
treasury_maturities4 = [1, 2, 3, 5, 7, 10, 20, 30]
treasury_maturities5 = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20]

for index, row in options.iterrows():
# The following code is complicated because there aren't data for some 
    # maturities and time periods.
    if pd.to_datetime("2004-01-02") <= row.QuoteDate <= pd.to_datetime("2018-10-15"):
        if pd.to_datetime("2004-01-02") <= row.QuoteDate <= pd.to_datetime("2006-02-08") and row.Time_to_Maturity > 25:
            list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities5])
            list_s = list_s + [40]
            differences.loc[len(differences)] = list_s
        elif (pd.to_datetime("2008-12-10") or pd.to_datetime("2008-12-18") or pd.to_datetime("2008-12-24")) == row.QuoteDate and 1.5/12 <= row.Time_to_Maturity <= 3.5/12:
            list_s = [0, 40, 40] # 40 is an arbitrary number bigger than 30
            list_s = list_s + [abs(maturity - row.Time_to_Maturity) for 
                                   maturity in treasury_maturities3]
            differences.loc[len(differences)] = list_s
            
        elif (pd.to_datetime("2008-12-10") or pd.to_datetime("2008-12-18") or pd.to_datetime("2008-12-24")) == row.QuoteDate and 3.5/12 < row.Time_to_Maturity <= 4.5/12:    
            list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
                           treasury_maturities2])
            list_s = list_s + [40, 40, 0]
            list_s = list_s + [abs(maturity - row.Time_to_Maturity) for 
                                   maturity in treasury_maturities4]
            differences.loc[len(differences)] = list_s
        elif pd.to_datetime("2010-10-11") == row.QuoteDate:
            if 1.5/12 <= row.Time_to_Maturity <= 2/12:
                list_s = [0, 40] # 40 is an arbitrary number bigger than 30
                list_s = list_s + [abs(maturity - (row + 1).Time_to_Maturity) 
                                    for maturity in treasury_maturities1]
                differences.loc[len(differences)] = list_s
            elif 2/12 < row.Time_to_Maturity <= 2.5/12:
                list_s = ([abs(maturity - (row + 1).Time_to_Maturity) for 
                            maturity in treasury_maturities2])
                list_s = list_s + [40, 0]
                list_s = list_s + [abs(maturity - (row + 1).Time_to_Maturity) 
                                    for maturity in treasury_maturities3]
                differences.loc[len(differences)] = list_s
        else:
            if 1.5/12 <= row.Time_to_Maturity <= 2/12:
                list_s = [0, 40] # 40 is an arbitrary number bigger than 30
                list_s = list_s + [abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities1]
                differences.loc[len(differences)] = list_s
            elif 2/12 < row.Time_to_Maturity <= 2.5/12:
                list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities2])
                list_s = list_s + [40, 0]
                list_s = list_s + [abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities3]
                differences.loc[len(differences)] = list_s
            else:
                list_s = [abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities]
                differences.loc[len(differences)] = list_s
    else:        
        list_s = [abs(maturity - row.Time_to_Maturity) for maturity in 
              treasury_maturities]
        differences.loc[len(differences)] = list_s

# Add to the options df, the columns for each Treasury maturity containing the 
    # differences calculated previously
options.reset_index(inplace = True, drop = True)
options = pd.concat([options, differences], axis = 1)

# Retrieve the maturity that is closest to the TTM of the option
maturity_closest_ttm = options[treasury_maturities].idxmin(axis = 1)

# Add maturity_closest_ttm as a new column in options df and change it's name.
options = pd.concat([options, maturity_closest_ttm], axis = 1)
options = options.rename(columns = {0:'Maturity_Closest_TTM'})

# Create list with the Treasury rate that matches the each option's QuoteDate 
    # and Maturity_Closest_TTM
rf_rate = []
for index, row in options.iterrows():
    rf_rate.append(float(treasury[row.Maturity_Closest_TTM].loc[(treasury["Date"] == row.QuoteDate)]))

# Add rf_rate as a column in the options df and drop unnecessary columns
options["RF_Rate"] = rf_rate
options = options.drop(treasury_maturities, axis = 1)
options = options.drop("Maturity_Closest_TTM", axis = 1)

# Create list with the standard deviations that match each option's QuoteDate
sigma_20 = []
for index, row in options.iterrows():
    sigma_20.append(float(underlying["Sigma_20_Days"].loc[underlying["Date"] == row.QuoteDate]))
 
# Add sigma_20 as a column in the options df
options["Sigma_20_Days"] = sigma_20

# Create list with the closing prices (of the underlying) that match each 
    # option's QuoteDate
underlying_price = []
for index, row in options.iterrows():
    underlying_price.append(float(underlying[" Close"].loc[underlying["Date"] == row.QuoteDate]))

# Add column of closing price of the underlying for each QuoteDate and drop
    # more unnecessary columns
options["Underlying_Price"] = underlying_price
options = options.drop(["expiration", "bid_eod", "ask_eod", "underlying_bid_eod",
                        "underlying_ask_eod"], axis = 1)

# Create csv file from the options df
options.to_csv('options-df.csv', index=False)
    
    