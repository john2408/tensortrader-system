import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels 
import pywt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


# linux: export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system/"
# win: ---

from tensortrader.transformations.denoising import *
from tensortrader.tasks.task_utils import create_logging
from tensortrader.constants import *

from datetime import datetime
import os
from pathlib import Path
import yaml

def main():

    CONF = yaml.safe_load(Path('../config/denoising.yml').read_text())

    # -----------------------------
    # Get config values
    # -----------------------------
    # Threshold for Wavelet Denoising (from 0.0 to 1.0)
    thresh = CONF['thresh']
    # Input and Output Paths
    input_data_path = CONF['input_data_path']
    output_data_path = CONF['output_data_path']
    #Denoising method 
    denoising_method = CONF['denoising_method']
    #Wavelet Function
    wavelet = CONF['wavelet']
    # Candle Stick Upsample Range (in minutes)
    minute_sampling = CONF['minute_sampling']
    # lookback subset to PACF pattern search
    pacf_days_subset = CONF['pacf_days_subset']
    # Number of hours to consider for 
    # lag pattern lookup
    nn_hours_pattern_lookup = CONF['nn_hours_pattern_lookup']
    # Length of Subset Timeseries
    subset_wavelet_transform = int(24*(60/minute_sampling)*pacf_days_subset) #60*hours # Number 
    # Number of lags to analyse in Partial Autocorrelation Function
    # 48 lags * 15 min_sampling / 60 hours per minute = 12 Hours in lags
    lags_pacf = (60/minute_sampling)*nn_hours_pattern_lookup  

    # Parameters Partial Autocorrelation Function
    alpha_pacf = CONF['alpha_pacf']
    method_pacf = CONF['method_pacf']

    run_name = f"return_denoising_{denoising_method}_{wavelet}"

    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M") 

    denoising_log_dir = os.path.join( Path(os.getcwd()).parents[0].parents[0],
                         'logs/denoising_logs',
                         f"Price_{timestamp}_{run_name}")

    if not os.path.exists(denoising_log_dir):
        os.mkdir(denoising_log_dir)

    with open(os.path.join(denoising_log_dir,'config.yml'), 'w') as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Return denoising log at", denoising_log_dir)

    LOG_FILENAME = os.path.join( denoising_log_dir,
                                f"{timestamp}_Denoising_return.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    logger.info(f"Max lag to analyse is: {lags_pacf} " )

    # -----------------------------
    # Data Import
    # -----------------------------
    df = pd.read_parquet(os.path.join(input_data_path, 
                        'Tensor_Portfolio.parquet'))

    # -----------------------------
    # Create Denoised dataframe
    # -----------------------------
    dfs = []

    for ticker in SYMBOLS:

        prices_return = df[df['Ticker'] == ticker]['Close_target_return_15m'].values

        # Get selected subset of data
        prices_return = prices_return[-subset_wavelet_transform:]

        # remove High Frequency Noise from Timeseries
        denoiser = denoising(signal = prices_return)

        if denoising_method == 'wavelet':
            denoised_prices_return = denoiser.wavelet_denoising( 
                                    thresh = thresh,
                                    wavelet = wavelet)


        # Get partial autocorrelation 
        pacf_values, confint = pacf(denoised_prices_return, 
                                    nlags = lags_pacf, 
                                    alpha = alpha_pacf, 
                                    method = method_pacf)

        # Get deepest significant lag level
        max_lag_pacf = get_significant_max_lag_pacf(pacf_values = pacf_values,
                                    confint = confint, 
                                    lags_pacf = lags_pacf )

        
        lag_analysis = f"""For Ticker {ticker}  
                        the max significant autocorrelation is 
                        {max_lag_pacf}  lag from {lags_pacf} lags analyzed
                        """    

        logger.info(lag_analysis)
        print(lag_analysis)                        

        df_temp = pd.DataFrame()
        df_temp['price_returns'] = prices_return
        df_temp['denoised_price_returns'] = denoised_prices_return
        df_temp['ticker'] = ticker
        df_temp['pacf_lag'] = int(max_lag_pacf)

        dfs.append(df_temp)
        del df_temp
        
    
    
    df_prices = pd.concat(dfs, ignore_index = True)

    logger.info("Storing Denoised prices to parquet")
    df_prices.to_parquet(os.path.join(output_data_path, "Tensor_Portfolio_denoised.parquet"))

    # -----------------------------
    # Create PDF with Plots
    # -----------------------------
    logger.info("Generating Denoised returns plots")
    generate_pdf_denoised_plots(df_prices = df_prices, 
                        storage_loc = denoising_log_dir, 
                        file_name = 'Tensor_Portfolio_denoised_prices')

    logger.info(f"denoised data sucessfully stored at {output_data_path} ")

if __name__ == "__main__":
    print("Denosing Price Return")
    main()
