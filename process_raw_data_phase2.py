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


# Load data
underlying = pd.read_csv("Processed data/underlying.csv")
treasury = pd.read_csv("Processed data/treasury.csv")
# treasury = pd.read_csv("Processed data/treasury.csv", keep_default_na=False, 
#                        na_values=['N/A '])
options = pd.read_csv("Processed data/options_phase1.csv")

# Convert dates to datetime64
options['QuoteDate'] = pd.to_datetime(options['QuoteDate'])
# options['Time_to_Maturity'] = pd.to_datetime(options['Time_to_Maturity'])
treasury["Date"] = pd.to_datetime(treasury["Date"])


"""
Get risk-free rates that match(or are close to) each option's maturity and 
QuoteDate
"""
"""Subtract the TTM of each option with the different maturities"""
# # List with the different maturities, in years, of Treasury bonds
# treasury_maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

# differences = pd.DataFrame(columns = treasury_maturities)
# treasury_maturities1 = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
# treasury_maturities2 = [1/12]
# treasury_maturities3 = [6/12, 1, 2, 3, 5, 7, 10, 20, 30]
# treasury_maturities4 = [1, 2, 3, 5, 7, 10, 20, 30]
# treasury_maturities5 = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20]
# date1 = pd.to_datetime("2004-01-02").to_datetime64()
# date2 = pd.to_datetime("2018-10-15").to_datetime64()
# date3 = pd.to_datetime("2006-02-08").to_datetime64()
# date4 = pd.to_datetime("2008-12-10").to_datetime64()
# date5 = pd.to_datetime("2008-12-18").to_datetime64()
# date6 = pd.to_datetime("2008-12-24").to_datetime64()


# Single-core attempt (too slow)
# for index, row in options.iterrows():
# for index, row in tqdm(options.iterrows()):
# # The following code is complicated because there aren't data for certain 
#     # maturities and time periods.
#     if date1 <= row.QuoteDate <= date2:
#         if date1 <= row.QuoteDate <= date3 and row.Time_to_Maturity > 25:
#             list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities5])
#             list_s = [list_s + [40]] # 40 is an arbitrary number bigger than 30
#             # differences.loc[len(differences)] = list_s
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True) 
#         elif (date4 or date5 or date6) == (row.QuoteDate and 1.5/12 <= 
#                                             row.Time_to_Maturity <= 3.5/12):
#             list_s = [0, 40, 40]
#             list_s = [list_s + [abs(maturity - row.Time_to_Maturity) for 
#                                     maturity in treasury_maturities3]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         elif (date4 or date5 or date6) == (row.QuoteDate and 3.5/12 < 
#                                             row.Time_to_Maturity <= 4.5/12):    
#             list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
#                             treasury_maturities2])
#             list_s = list_s + [40, 40, 0]
#             list_s = [list_s + [abs(maturity - row.Time_to_Maturity) for 
#                                     maturity in treasury_maturities4]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         else:
#             if 1.5/12 <= row.Time_to_Maturity <= 2/12:
#                 list_s = [0, 40]
#                 list_s = [list_s + [abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities1]]
#                 # differences.loc[len(differences)] = list_s
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             elif 2/12 < row.Time_to_Maturity <= 2.5/12:
#                 list_s = ([abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities2])
#                 list_s = list_s + [40, 0]
#                 list_s = [list_s + [abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities3]]
#                 # differences.loc[len(differences)] = list_s
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             else:
#                 list_s = [[abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities]]
#                 # differences.loc[len(differences)] = list_s
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#     else:        
#         list_s = [[abs(maturity - row.Time_to_Maturity) for maturity in 
#               treasury_maturities]]
#         # differences.loc[len(differences)] = list_s
#         differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)


# DataFrame.apply attempt #1
# def sub_mat(qd, ttm):
#     global differences
    
#     # The following code is complicated because there aren't data for certain 
#     # maturities and time periods.
#     if date1 <= qd <= date2:
#         if date1 <= qd <= date3 and ttm > 25:
#             list_s = ([abs(maturity - ttm) for maturity in 
#               treasury_maturities5])
#             list_s = [list_s + [40]] # 40 is an arbitrary number bigger than 30
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True) 
#         elif (date4 or date5 or date6) == (qd and 1.5/12 <= 
#                                             ttm <= 3.5/12):
#             list_s = [0, 40, 40]
#             list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities3]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         elif (date4 or date5 or date6) == (qd and 3.5/12 < 
#                                             ttm <= 4.5/12):    
#             list_s = ([abs(maturity - ttm) for maturity in 
#                             treasury_maturities2])
#             list_s = list_s + [40, 40, 0]
#             list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities4]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         else:
#             if 1.5/12 <= ttm <= 2/12:
#                 list_s = [0, 40]
#                 list_s = [list_s + [abs(maturity - ttm) 
#                                     for maturity in treasury_maturities1]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             elif 2/12 < ttm <= 2.5/12:
#                 list_s = ([abs(maturity - ttm) for maturity 
#                             in treasury_maturities2])
#                 list_s = list_s + [40, 0]
#                 list_s = [list_s + [abs(maturity - ttm) for maturity in 
#               treasury_maturities3]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             else:
#                 list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#     else:        
#         list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#         differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
    
#     return differences


# DataFrame.apply attempt #2
# def sub_mat(qd, ttm):
#     global differences
    
#     with tqdm() as ttm:
#         # The following code is complicated because there aren't data for certain 
#             # maturities and time periods.
#         if date1 <= qd <= date2:
#             if date1 <= qd <= date3 and ttm > 25:
#                 list_s = ([abs(maturity - ttm) for maturity in 
#                            treasury_maturities5])
#                 list_s = [list_s + [40]] # 40 is an arbitrary number bigger than 30
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True) 
#             elif (date4 or date5 or date6) == (qd and 1.5/12 <= 
#                                             ttm <= 3.5/12):
#                 list_s = [0, 40, 40]
#                 list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities3]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             elif (date4 or date5 or date6) == (qd and 3.5/12 < 
#                                             ttm <= 4.5/12):    
#                 list_s = ([abs(maturity - ttm) for maturity in 
#                             treasury_maturities2])
#                 list_s = list_s + [40, 40, 0]
#                 list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities4]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             else:
#                 if 1.5/12 <= ttm <= 2/12:
#                     list_s = [0, 40]
#                     list_s = [list_s + [abs(maturity - ttm) 
#                                     for maturity in treasury_maturities1]]
#                     differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#                 elif 2/12 < ttm <= 2.5/12:
#                     list_s = ([abs(maturity - ttm) for maturity 
#                             in treasury_maturities2])
#                     list_s = list_s + [40, 0]
#                     list_s = [list_s + [abs(maturity - ttm) for maturity in 
#               treasury_maturities3]]
#                     differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#                 else:
#                     list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#                     differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         else:        
#             list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
    
#     return differences


# options.apply(lambda options: sub_mat(options["QuoteDate"], 
#                                 options["Time_to_Maturity"]), axis = 1)


# Multi-core processing attempt (ran into memory/RAM problem)
# def ttm_subtraction_maturity(ttm, qd):
#     """The following code is complicated because there aren't data for certain 
# maturities and time periods."""
#     global differences

#     if date1 <= qd <= date2:
#         if date1 <= qd <= date3 and ttm > 25:
#             list_s = ([abs(maturity - ttm) for maturity in 
#               treasury_maturities5])
#             list_s = [list_s + [40]] # 40 is an arbitrary number bigger than 30
#             # differences.loc[len(differences)] = list_s
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True) 
#         elif (date4 or date5 or date6) == (qd and 1.5/12 <= 
#                                             row.Time_to_Maturity <= 3.5/12):
#             list_s = [0, 40, 40]
#             list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities3]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         elif (date4 or date5 or date6) == (qd and 3.5/12 < 
#                                             ttm <= 4.5/12):    
#             list_s = ([abs(maturity - ttm) for maturity in 
#                             treasury_maturities2])
#             list_s = list_s + [40, 40, 0]
#             list_s = [list_s + [abs(maturity - ttm) for 
#                                     maturity in treasury_maturities4]]
#             differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#         else:
#             if 1.5/12 <= ttm <= 2/12:
#                 list_s = [0, 40]
#                 list_s = [list_s + [abs(maturity - ttm) for maturity in 
#               treasury_maturities1]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             elif 2/12 < ttm <= 2.5/12:
#                 list_s = ([abs(maturity - ttm) for maturity in 
#               treasury_maturities2])
#                 list_s = list_s + [40, 0]
#                 list_s = [list_s + [abs(maturity - ttm) for maturity in 
#               treasury_maturities3]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#             else:
#                 list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#                 differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
#     else:        
#         list_s = [[abs(maturity - ttm) for maturity in 
#               treasury_maturities]]
#         differences = differences.append(pd.DataFrame(list_s, 
#                         columns = treasury_maturities), ignore_index = True)
    
#     return differences
    
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     differences_list = list(executor.map(ttm_subtraction_maturity, 
#                     tqdm(options["Time_to_Maturity"]), options["QuoteDate"]))
    
# differences = pd.concat(tqdm(differences_list))


# # Add the columns for each Treasury maturity, containing the 
#     # differences calculated previously, to the options df.
# options.reset_index(inplace = True, drop = True)
# options = pd.concat([options, differences], axis = 1)

# # Retrieve the maturity that is closest to the TTM of the option
# maturity_closest_ttm = options[treasury_maturities].idxmin(axis = 1)

# # Add maturity_closest_ttm as a new column in options df and change it's name.
# options = pd.concat([options, maturity_closest_ttm], axis = 1)
# options = options.rename(columns = {0:'Maturity_Closest_TTM'})

# # Create list with the Treasury rates that matches each option's QuoteDate 
#     # and Maturity_Closest_TTM
# rf_rate = []
# # for index, row in options.iterrows():
# for index, row in tqdm(options.iterrows()):
#     (rf_rate.append(float(treasury[row.Maturity_Closest_TTM].loc[(treasury["Date"] 
#                                                         == row.QuoteDate)])))

"""
Get a 3-month risk-free rate that match each option's QuoteDate
"""
"""Create list with the 3-month Treasury rates that match each option's 
QuoteDate"""
print("\n" + f"Total number of options: {options.shape[0]}")

rf_rate = []
QuoteDate_df = options[['QuoteDate']]
QuoteDate_df = (QuoteDate_df.assign(in_treasury = 
                                QuoteDate_df.QuoteDate.isin(treasury.Date)))

# for index, row in tqdm(QuoteDate_df.iterrows()):
#     (rf_rate.append(float(treasury["0.25"].loc[(treasury["Date"] == 
#                                                 row.QuoteDate)])))

for index, row in tqdm(QuoteDate_df.iterrows()):
# for index, row in tqdm(options.iterrows()):
    if row.in_treasury == True:
        (rf_rate.append(float(treasury["0.25"].loc[(treasury["Date"] 
                                                        == row.QuoteDate)])))
    else:
        """This situation happens if the treasury df doesn't have a rate for a 
    date in which options exist. -4 is an arbitrary negative number"""
        rf_rate.append(-4)

"""Add rf_rate as a column in the options df and drop unnecessary columns"""
options["RF_Rate"] = rf_rate
# options = options.drop(treasury_maturities, axis = 1)
# options = options.drop("Maturity_Closest_TTM", axis = 1)

# Remove options with rf rate = -4
indices = options[options['RF_Rate'] == -4].index
options.drop(indices ,inplace = True)

# Create csv file from the options df
options.to_csv('Processed data/options_phase2.csv', index = False)

