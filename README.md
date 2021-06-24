INSTRUCTIONS ARE OUTDATED AND WILL BE REVISED AT THE END OF THE PROJECT. Files with "SO" in the name are for StackOverflow questions.

# SPX_Options_ANN
Public repository for the code related to my Master's thesis. My e-mail is: dpintof@disroot.org

Work related to the paper Ke and Yang (2019) Option Pricing with Deep Learning. My code is inspired by what the authors made publicly available: https://github.com/ycm/cs230-proj. The objective is to price/evaluate vanilla options (financial derivatives) using different kinds of neural networks. 

Instructions:

Process_raw_data1.py is to be used with the full proprietary dataset. Process_raw_data2.py is to be used with the small free dataset.

1. Run Process_raw_data1.py to create options-df.csv and underlying_df.csv in the "Processed data" folder. It uses data in the "Raw data" folder.
I did not include the database used in this file because it is proprietary. There are comments in the code on how to test it with a small sample.

OR

1. Run Process_raw_data2.py to create options-df.csv and underlying_df.csv in the "Processed data" folder. It uses data in the "Raw data" folder. Because some data files are quite large I don't include them in the repository. There are comments in the code on how to run it with the data in the repository (a small sample). The sources of the free raw data are:
 * Underlying - https://www.wsj.com/market-data/quotes/index/SPX/historical-prices
 * Options (September 2018 and August 2019, respectively) - https://www.historicaloptiondata.com/files/Batch_PRO_Sample_PHC.zip and https://www.historicaloptiondata.com/files/Sample_L2_2019_August.zip
 * Treasury Rates - https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2019  

2. MLP1 model (what is explained here also applies to other models): 

    2.1. Run mlp1_call.py and mlp1_put.py. Uses the data in the "Processed data" folder.

    2.2. Run mlp1_error_metrics.py. It returns train-MSE (the mean squared error on the training set) and metrics on the test set: MSE, Bias (the median percent error), AAPE (the average absolute percent error), MAPE (the median absolute percent error), and PEX% (the percentage of observations within ±X% of the actual price).   
 
3. MLP2 model: 
        
    3.1. Run mlp2_call.py and mlp2_put.py.
       		
    3.2. Run mlp2_error_metrics.py

4. LSTM model:
        
    4.1. Run lstm.py

    4.2. Run lstm_error_metrics.py

5. Black-Scholes-Merton (BSM) model:

    5.1. Run bsm.py

    5.2. Run bsm_error_metrics.py

6. Create graphs showing training and validation losses over epoches:

    6.1. Run ann_loss_graphs.py

