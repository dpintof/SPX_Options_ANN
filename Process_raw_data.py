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
underlying1 = pd.read_csv("Data/Underlying/SPX_August-September_2018.csv")

# Create df for the data of the underlying from July to August 2019
underlying2 = pd.read_csv("Data/Underlying/SPX_July-August_2019.csv")

# Create df with all underlying data
underlying = underlying1.append(underlying2)

# Remove unnecessary columns
underlying = underlying.drop([" Open", " High", " Low"], axis = 1)

# Creates a new column with the standard deviation of the returns of the past 20 days
underlying['Historical_Vol'] = underlying[" Close"].rolling(21).apply(lambda x:
                                                (np.diff(x) / x[:-1]).std())


# Treasury rates
# Create df for treasury rates from August to September 2018
treasury1 = pd.read_csv("Data/Treasury/Treasury_rates_August-September_2018.csv")

# Create df for treasury rates from July to August 2019
treasury2 = pd.read_csv("Data/Treasury/Treasury_rates_July-August_2019.csv")

# Create df with all treasury rates data
treasury = treasury1.append(treasury2)


# Options
# Create df for September 2018 options data
options1 = pd.read_csv("Data/Options/SPX_20180904_to_20180928.csv")

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
p = Path("Data/Options") 

# Create a list of the option files from August 2019
options2_files = list(p.glob('L2_options_201908*.csv'))

# Creates df from all files (might take a while)
# options2 = pd.concat([pd.read_csv(f) for f in options2_files]) # REMOVE COMMENT IN FINAL VERSION
options2 = pd.read_csv("Data/Options/L2_options_20190801.csv")

# Deletes rows for options with an underlying asset that isn't SPX
options2 = options2.loc[options2['UnderlyingSymbol'] == "SPX"]

# Remove unnecessary columns
options2 = options2.drop(['UnderlyingSymbol', 'UnderlyingPrice', 'Exchange', 
                          'OptionSymbol', 'OptionExt', 'Last', 'Volume', 
                          'OpenInterest', 'IV', 'Delta', 'Gamma', 'Theta', 
                          'Vega', 'AKA'], axis = 1)

# Rename columns DataDate to QuoteDate and Type to OptionType
options2 = options2.rename(columns = {'DataDate': 'QuoteDate', "Type": "OptionType"})

# Create df with all options data
options = options1.append(options2)
options = options.iloc[:5] # REMOVE IN FINAL VERSION

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

# List with the different maturities, in years, of Treasury bonds
treasury_maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30] 
# treasury_maturities = [1/12, 2/12] # REMOVE COMMENT IN FINAL VERSION

# Subtract the TTM of each option with the different maturities
differences = pd.DataFrame(columns = treasury_maturities)

for index, row in options.iterrows():
    list_s = [abs(maturity - row.Time_to_Maturity) for maturity in 
          treasury_maturities]
    differences.loc[len(differences)] = list_s

# Add Treasury maturity columns (the differences df) to the options df
options.reset_index(inplace = True, drop = True)
options = pd.concat([options, differences], axis=1)


# transformar o nome das colunas de Treasury nos valores de treasury_maturities?
# subtrair ttm de cada opção às diferentes maturidades das obrigações
# aplicar abs a cada subtração
# selecionar o menor valor absoluto
# relacionar esse mínimo com a maturidade da obrigação, tendo atenção aos NAs
# recolher a taxa respetiva a essa maturidade e à data da opção em causa
# adicionar nova coluna a options com a rf rate para cada opção

# create new column in options for the vol(sigma)
# get vol from the underlying df for the corresponding DataDate of each option

# Add column with the standard deviation calculated for the underlying
# options["Historical_Vol"] = underlying[underlying["Date"] == 
#                                        options["QuoteDate"]]["Historical_Vol"]

# Create csv file from the options df
# options.to_csv('options-df.csv', index=False)
