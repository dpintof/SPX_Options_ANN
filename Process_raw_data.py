#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:07:34 2021

@author: Diogo
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np


# Underlying
# Create dataframe (df) for the data of the underlying from August to September 2018
underlying1 = pd.read_csv("Raw data/Underlying/SPX_August-September_2018.csv")

# Create df for the data of the underlying from July to August 2019
underlying2 = pd.read_csv("Raw Data/Underlying/SPX_July-August_2019.csv")

# Convert dates on Date columns to datetime64
underlying1['Date'] = pd.to_datetime(underlying1['Date'])
underlying2['Date'] = pd.to_datetime(underlying2['Date'])

# Sort underlying df by Date column, in ascending order
underlying1 = underlying1.sort_values(by='Date')
underlying2 = underlying2.sort_values(by='Date')

# Creates new column with the standard deviation of the returns from the past 20 days
underlying1['Historical_Vol'] = underlying1[" Close"].rolling(20).apply(lambda x:
                                                (np.diff(x) / x[:-1]).std())
underlying2['Historical_Vol'] = underlying2[" Close"].rolling(20).apply(lambda x:
                                                (np.diff(x) / x[:-1]).std())

# Create df with all underlying data
underlying = underlying1.append(underlying2)

# Remove unnecessary columns
underlying = underlying.drop([" Open", " High", " Low"], axis = 1)

# Create csv file from the underlying df
underlying.to_csv('underlying_df.csv', index=False)

    
# Treasury rates
# Create df for treasury rates from August to September 2018
treasury1 = pd.read_csv("Raw data/Treasury/Treasury_rates_August-September_2018.csv")

# Create df for treasury rates from July to August 2019
treasury2 = pd.read_csv("Raw data/Treasury/Treasury_rates_July-August_2019.csv")

# Create df with all treasury rates data
treasury = treasury1.append(treasury2)

# List with the different maturities, in years, of Treasury bonds
treasury_maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30] 

# Change column names of treasury df
treasury_columns = ["Date", 1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30] 
treasury.columns = treasury_columns

# Convert data on Date column of treasury df to datetime64
treasury['Date'] = pd.to_datetime(treasury['Date'])


# Options
# Create df for September 2018 options data
options1 = pd.read_csv("Raw data/Options/SPX_20180904_to_20180928.csv")

# Remove unnecessary columns
options1 = options1.drop(["DNID", 'UnderlyingSymbol', 'UnderlyingPrice', 
                          'OptionSymbol', 'OpenPrice', 'HighPrice', 'LowPrice',
                          'LastPrice', 'Volume', 'TradeCount', 'OpenInterest',
                          'T1OpenInterest', 'IVGreeks', 'IVBid', 'IVAsk',
                          'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Vanna', 
                          'Vomma', 'Charm', 'DvegaDtime', 'Color', 'Speed',
                          'TradeSize', 'TradeDateTime', 'TradedMarket',
                          'BidMarket', 'AskMarket', 'BidSize', 'AskSize', 
                          'OptionAlias'], axis = 1)

# Set the path to files for options from August 2019
p = Path("Raw data/Options")

# Create a list of the option files from August 2019
options2_files = list(p.glob('L2_options_201908*.csv'))

# Creates df from all files
options2 = pd.concat([pd.read_csv(f) for f in options2_files]) 
# options2 = pd.read_csv("Raw data/Options/L2_options_20190801.csv") # FOR TESTING 
#     THE CODE WITH A SMALL SAMPLE

# Deletes rows for options with an underlying asset that isn't SPX or SPXW
options2 = options2.loc[options2['UnderlyingSymbol'].isin(["SPX", "SPXW"])]


# Remove unnecessary columns
options2 = options2.drop(['UnderlyingSymbol', 'UnderlyingPrice', 'Exchange', 
                          'OptionSymbol', 'OptionExt', 'Last', 'Volume', 
                          'OpenInterest', 'IV', 'Delta', 'Gamma', 'Theta', 
                          'Vega', 'AKA'], axis = 1)

# Rename columns DataDate to QuoteDate and Type to OptionType
options2 = options2.rename(columns = {'DataDate': 'QuoteDate', "Type": "OptionType"})

# Create df with all options data
options = options1.append(options2)
# options = options1 # FOR TESTING THE CODE WITH A SMALL SAMPLE
# options = options.iloc[1052:1546] # FOR TESTING THE CODE WITH A SMALL SAMPLE 
    # FOR WHICH THE TIME TO MATURITY IS CLOSE TO 2 YEARS. WE DON'T HAVE 2 YEAR
    # TREASURY RATES FOR SOME DATES.

# Create function that returns the number of years between two dates
def years_between(d1, d2):
    d1 = datetime.strptime(d1, "%m/%d/%Y")
    d2 = datetime.strptime(d2, "%m/%d/%Y")
    return abs((d2 - d1).days / 365)

# Calculate the time to maturity (TTM), in years, for each option
ttm = []

for index, row in options.iterrows():
    d1 = row.Expiration
    d2 = row.QuoteDate
    d = years_between(d1, d2)
    ttm.append(d)

# Create new column with the TTM
options['Time_to_Maturity'] = ttm

# Calculate the average of the bid and ask prices
average_price = []

for index, row in options.iterrows():
    bid = row.Bid
    ask = row.Ask
    average = (ask + bid) / 2
    average_price.append(average)
    
# Create new column with the average price
options['Average_Price'] = average_price

# Convert data on QuoteDate column of options df to datetime64
options['QuoteDate'] = pd.to_datetime(options['QuoteDate'])

# Subtract the TTM of each option with the different maturities
differences = pd.DataFrame(columns = treasury_maturities)
treasury_maturities1 = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
treasury_maturities2 = [1/12]
treasury_maturities3 = [6/12, 1, 2, 3, 5, 7, 10, 20, 30]

for index, row in options.iterrows():
# The following code gets complicated because there aren't data for 2 month 
    # rates between 2018-08-01 and 2018-09-28. There's probably a much cleaner 
    # way of doing this.
    if pd.to_datetime("2018-08-01") <= row.QuoteDate <= pd.to_datetime("2018-09-28"):
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
hist_vol = []
for index, row in options.iterrows():
    hist_vol.append(float(underlying["Historical_Vol"].loc[underlying["Date"] == row.QuoteDate]))
 
# Add hist_vol as a column in the options df
options["Historical_Vol"] = hist_vol

# Create list with the closing prices (of the underlying) that match each option's QuoteDate
underlying_price = []
for index, row in options.iterrows():
    underlying_price.append(float(underlying[" Close"].loc[underlying["Date"] == row.QuoteDate]))

# Add column of closing price of the underlying for each QuoteDate and drop
    # more unnecessary columns
options["Underlying_Price"] = underlying_price
options = options.drop(["Expiration"], axis = 1)

# Create csv file from the options df
options.to_csv('options-df.csv', index=False)

