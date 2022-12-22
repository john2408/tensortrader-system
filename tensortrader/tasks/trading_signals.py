import yaml
from pathlib import Path
import joblib
import keras as ks
import pandas as pd
import os


import warnings
warnings.filterwarnings("ignore")

from tensortrader.constants import *
from tensortrader.tasks.task_utils import create_logging
from tensortrader.transformations.denoising import *

import datetime

# linux: export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system/"
# win: ---


def main():

    # -----------------------------
    # Get config values
    # -----------------------------
    CONF = yaml.safe_load(Path('../config/tcn_training.yml').read_text())

    # Candle Stick Upsample Range (in minutes)
    minute_sampling = CONF['minute_sampling']
    # lookback subset to PACF pattern search
    pacf_days_subset = CONF['pacf_days_subset']
    # Length of Subset Timeseries
    subset_wavelet_transform = int(24*(60/minute_sampling)*pacf_days_subset) #60*hours # Number 

    # Input data Loc - historical return prices
    input_data_path_price_return = CONF['input_data_path_price_return']
    db_name_price_return = CONF['db_name_price_return']
    
    # Input data Loc - historical denoised return prices
    input_data_path_denoised_return = CONF['input_data_path_denoised_return']
    db_name_denoised_return = CONF['db_name_denoised_return']

    # Model Location
    model_loc = CONF['model_loc']

    # Temporal Convolutatin Networks Params
    n_features = CONF['n_features']

    # Denoising Method Parameters
    denoising_method = CONF['denoising_method']
    thresh = CONF['thresh']
    wavelet = CONF['wavelet']

    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") 

    signal_log_dir = os.path.join( Path(os.getcwd()).parents[0].parents[0],
                         'logs/trading_signal_logs',
                         f"Trading_signal_{timestamp}")

    if not os.path.exists(signal_log_dir):
        os.mkdir(signal_log_dir)

    with open(os.path.join(signal_log_dir,'config.yml'), 'w') as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Trading singal log at", signal_log_dir)

    LOG_FILENAME = os.path.join( signal_log_dir,
                                f"{timestamp}_Trading_Signal.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Data Load
    # -----------------------------

    # Latest Prices returns 
    try:
        logger.info("Reading latest price returns")
        path_loc = os.path.join(input_data_path_price_return, db_name_price_return)
        df = pd.read_parquet(path_loc)
    except Exception as e:
        print(f"Erorr {e}")
        logger.info(f"{e}")

    # Denoised prices
    try:
        logger.info("Reading latest denoised prices and PACF")
        path_loc = os.path.join(input_data_path_denoised_return, db_name_denoised_return)
        df_prices = pd.read_parquet(path_loc)
    except Exception as e:
        print(f"Erorr {e}")
        logger.info(f"{e}")

    # -----------------------------
    # Generate Signal
    # -----------------------------
    for ticker in SYMBOLS:

        print("Getting Predicition for ticker", ticker)
        logger.info(f"Getting Predicition for ticker {ticker}")
        
        try:
            # load NN Model
            filepath = os.path.join(model_loc, 'models', f'TCN_Model_{ticker}')
            print("Reading model at loc", filepath)
            model = ks.models.load_model(filepath)
        except Exception as e:
            print(f"Erorr {e}")
            logger.info(f"{e}")
        
        try:
            # load Scaler
            filepath = os.path.join(model_loc, 'scalers', f'Scaler_{ticker}.pkl')
            print("Reading model at loc", filepath)
            scaler = joblib.load(filepath)
        except Exception as e:
            print(f"Erorr {e}")
            logger.info(f"{e}")
                

        # Get last prices
        # (1) Database with 15m candle bars
        prices_return = df[df['Ticker'] == ticker]['Close_target_return_15m'].values

        # Get selected subset of data
        # (2) Calculate price return
        prices_return = prices_return[-subset_wavelet_transform:]

        # (3) Remove High Frequency Noise from Timeseries
        denoiser = denoising(signal = prices_return)

        if denoising_method == 'wavelet':
            denoised_prices_return = denoiser.wavelet_denoising( 
                                    thresh = thresh,
                                    wavelet = wavelet)

        # PACF Lag
        # Step 0. From Model Training
        pacf_lag = df_prices[df_prices['ticker'] == ticker]['pacf_lag'].values[0]

        # (4) Get Signal Prediction
        # Get input data for model
        batch = denoised_prices_return[-pacf_lag:]

        print("PACF lag for ticker", ticker, " is: ", pacf_lag)
        logger.info(f"PACF lag for ticker {ticker} is {pacf_lag}")

        # Adjust Tensor Input shape
        batch = batch.reshape(1, pacf_lag, n_features)

        # Return prediction
        model_prediction = model.predict(batch)
        return_prediction = scaler.inverse_transform(model_prediction).reshape(1,-1)[0]

        # From ReturnSignal Class
        # ------------------------
        # r_nth(t+n) = ((P(t+n)/P(t)) - 1)**(1/n)  - 1
        #
        # 
        # --------------------------------------------
        # r_nth(t+n) +1 = ((P(t+n)/P(t)) - 1) **(1/n)
        # (r_nth(t+n) +1)**(n) = (P(t+n)/P(t)) - 1)
        # ((r_nth(t+n) +1)**(n)) + 1 = (P(t+n)/P(t)) 
        # P(t+n) = (((r_nth(t+n) +1)**(n)) + 1 )/ P(t) 

        if return_prediction > 0:
            print("Going Long")
        elif return_prediction < 0:
            print("going Short")
        else:
            print("Going neutral")
        
        del model
        del scaler

if __name__ == "__main__":
    print("Generating Trading Signal")
    main()
