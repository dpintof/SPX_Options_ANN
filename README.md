# SPX_Options_ANN
Public repository for the code related to my Master's thesis.

Work related to the paper Ke and Yang (2019) Option Pricing with Deep Learning. My code is inspired by what the authors made publicly available: https://github.com/ycm/cs230-proj.

Instructions:
    1. Run Process_raw_data2.py to create options-df.csv and underlying_df.csv in the "Processed data" folder. It uses data in the "Raw data" folder.
       I recommended using a small sample for testing the data as the current database is huge and takes a long time to process. For a test with a small sample comment line 63 and uncomment line 65.
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
        6.1. Run ann_loss_graphs.py (still not finished)



				

			

		

	
