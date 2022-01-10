# SPX_Options_ANN
Public repository for the code related to my Master's thesis. My e-mail is: dpintof@disroot.org

Work related to the paper Ke and Yang (2019) Option Pricing with Deep Learning. My code is inspired by what the authors made publicly available: https://github.com/ycm/cs230-proj. The objective is to price/evaluate vanilla options (financial derivatives) using different kinds of neural networks. 

# Instructions
process_raw_data_phase1.py, process_raw_data_phase2.py and process_raw_data_phase3_final.py were used with the full proprietary dataset. 
process_raw_free_dataset.py is to be used with the small free dataset.

1. Run process_raw_data_phase1.py, process_raw_data_phase2.py and process_raw_data_phase3_final.py sequentially to create options-df.csv and underlying_df.csv in the "Processed data" folder. It uses data in the "Raw data" folder.

Or

1. Run process_raw_free_dataset.py to create options-df.csv and underlying_df.csv in the "Processed data" folder. It uses data in the "Raw data" folder. Because some data files are quite large I don't include them in the repository. The sources of the free raw data are:
 * Underlying - https://www.wsj.com/market-data/quotes/index/SPX/historical-prices
 * Options (September 2018 and August 2019, respectively) - https://www.historicaloptiondata.com/files/Batch_PRO_Sample_PHC.zip and https://www.historicaloptiondata.com/files/Sample_L2_2019_August.zip
 * Treasury Rates - https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2019

(...) not finished.
