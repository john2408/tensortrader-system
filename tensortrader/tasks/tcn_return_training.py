import datetime
from pathlib import Path

import joblib
import keras as ks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_percentage_error
from tcn import TCN
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensortrader.constants import *
from tensortrader.ML.dl_models import tcn_model
from tensortrader.tasks.task_utils import create_logging
from tensortrader.transformations.denoising import *

#export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system"

def main():

    # -----------------------------
    # Get config values
    # -----------------------------
    path = "/mnt/d/Tensor/tensortrader-system/tensortrader/config/tcn_training.yml"
    #path = "../config/tcn_training.yml"
    CONF = yaml.safe_load(Path(path).read_text())

    # Input data Loc - historical denoised return prices
    input_data_path_denoised_return = CONF['input_data_path_denoised_return']
    db_name_denoised_return = CONF['db_name_denoised_return']

    # Model Location
    model_storage_loc = CONF['model_storage_loc']

    # Temporal Convolution Networks Params
    n_features = CONF['n_features']
    batch_size = CONF['batch_size']
    epochs = CONF['epochs']
    verbose = CONF['verbose']
    test_size = CONF['test_size'] # as a percentage 0.2 -> 20%
    seed = CONF['seed']
    dilations = CONF['dilations']
    kernel_size = CONF['kernel_size']
    monitor = CONF['monitor']
    patience = CONF['patience']
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    logs_folder = CONF['logs_folder']

    # -----------------------------
    # Logging Config
    # -----------------------------

    print("Storing Training logs at", logs_folder)

    LOG_FILENAME = os.path.join( logs_folder,
                                f"{timestamp}_Training_ML_return.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Model Storage Location
    # -----------------------------

    model_dir = os.path.join( model_storage_loc,  f"{timestamp}_TCN_Training")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open(os.path.join(model_dir,'config.yml'), 'w') as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Trading Models at", model_dir)

    # -----------------------------
    # Load Denoised Price Returns
    # -----------------------------

    # Denoised prices
    try:
        logger.info("Reading latest denoised prices and PACF")
        path_loc = os.path.join(input_data_path_denoised_return, db_name_denoised_return)
        df_prices = pd.read_parquet(path_loc)
    except Exception as e:
        print(f"Erorr {e}")
        logger.info(f"{e}")


    # ------------------------------------
    # Train Temporal Convolutional Networks
    # -------------------------------------
    dfs_train = []
    dfs_test = []
    dfs_metric = []

    #SYMBOLS = ['BTCUSDT']
    logger.info("Training Temporal Convolutional Networks")
    ticker_pacf_lags = {}

    for ticker in SYMBOLS:

        logger.info(f"\tTraining model for ticker {ticker}")
        print("\nTraining model for ticker ", ticker)

        logger.info(f"\tGetting Denoised Return price data for ticker: {ticker}")
        df_temp = df_prices[df_prices['ticker'] == ticker].copy()

        # Input Denoised timeseries of price returns
        ts = df_temp['denoised_price_returns'].values.reshape(-1, 1)

        # Input timestamps
        timestamps = df_temp['timestamp_local']

        print("Length ts:", len(ts))
        print("Length timestamps:", len(timestamps))

        lag_length = df_temp['pacf_lag'].values[0]

        ticker_pacf_lags[ticker] = lag_length

        logger.info(f"\tLag length PACF :  {lag_length}")

        logger.info(f"\tTraining TCN Model")
        ticker_tcn_model = tcn_model(
                                ts_data = ts,
                                timestamps = timestamps,
                                test_size = test_size,
                                lag_length = lag_length,
                                n_features = n_features,
                                seed = seed,
                                dilations = dilations,
                                kernel_size = kernel_size,
                                epochs = epochs,
                                patience = patience,
                                monitor = monitor,
                                verbose  = verbose)

        # Fit TCN Model
        ticker_tcn_model.fit()

        # Train Test Batches
        X_train, Y_train, X_test, Y_test = ticker_tcn_model.test_train_batches

        # Train/Test Timestamps
        timestamps_train, timestamps_test = ticker_tcn_model.train_test_timestamps

        forecast_train = ticker_tcn_model.scaler.inverse_transform(ticker_tcn_model.model.predict(X_train)).reshape(1,-1)[0]
        forecast_test = ticker_tcn_model.scaler.inverse_transform(ticker_tcn_model.model.predict(X_test)).reshape(1,-1)[0]

        original_train = ticker_tcn_model.scaler.inverse_transform(Y_train).reshape(1,-1)[0]
        original_test = ticker_tcn_model.scaler.inverse_transform(Y_test).reshape(1,-1)[0]

        print("len timestamps_test,", len(timestamps_test))
        print("len original test", len(original_test))

        df_train = pd.DataFrame()
        df_train['timestamp'] = timestamps_train
        df_train['forecast_train'] = forecast_train
        df_train['original_train'] = original_train
        df_train['ticker'] = ticker

        df_test = pd.DataFrame()
        df_test['timestamp'] = timestamps_test
        df_test['forecast_test'] = forecast_test
        df_test['original_test'] = original_test
        df_test['ticker'] = ticker

        dfs_train.append(df_train)
        dfs_test.append(df_test)

        pearson_coff = np.corrcoef(forecast_test, original_test)[0,1]
        pearson_coff = np.round(pearson_coff, 3)

        logger.info("\tPearson Coefficient is {}".format(np.round(pearson_coff, 3)))
        print("Pearson coefficient is: " , np.round(pearson_coff, 3), "for one step ahead forecast" )

        df_metric = pd.DataFrame({'ticker' : [ticker],
                                'pearson_coff': [pearson_coff] })

        dfs_metric.append(df_metric)

        # Store File
        print("Storing Model")
        logger.info("\tStoring Keras Model")
        filepath = os.path.join(model_dir, f'TCN_Model_{ticker}')
        ticker_tcn_model.model.save(filepath)

        # Store Scaler
        print("Storing Scaler")
        logger.info("\tStoring Standard Scaler")
        filepath = os.path.join(model_dir, f'Scaler_{ticker}.pkl')
        joblib.dump(ticker_tcn_model.scaler, filepath)

        del df_test
        del df_train
        del ticker_tcn_model

    # ------------------------------------
    # Storing Dict with PACF Lags
    # -------------------------------------
    logger.info("\tStoring PACF Lags")
    filepath = os.path.join(model_dir, f'PACF_lags.pkl')
    joblib.dump(ticker_pacf_lags, filepath)


    # ------------------------------------
    # Training Results
    # -------------------------------------
    df_train_data = pd.concat(dfs_train, ignore_index = True)
    df_test_data = pd.concat(dfs_test, ignore_index = True)
    df_metric_data = pd.concat(dfs_metric, ignore_index=True)

    logger.info("\tTraining Results")
    filepath = os.path.join(model_dir, f'Metric_Results.csv')
    df_metric_data.to_csv(filepath)

    logger.info("\tTest Results")
    filepath = os.path.join(model_dir, f'Test_Results.parquet')
    df_test_data.to_parquet(filepath)

    print(df_metric_data)

    # ------------------------------------
    # Train/Test Plots
    # -------------------------------------
    logger.info("\tGenerating Train/Test Plots")

    file_name = "Test_Train_plots"
    pp = PdfPages(os.path.join(model_dir, f"{file_name}.pdf"))

    for ticker in df_metric_data['ticker'].unique():

        df_tmp = df_test_data[df_test_data['ticker'] == ticker].copy()
        timestamps = df_tmp['timestamp']
        df_tmp.set_index('timestamp', inplace = True)
        original_test = df_tmp['original_test']
        forecast_test = df_tmp['forecast_test']

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(original_test, color="b", alpha=0.99, label='original')
        ax.plot(forecast_test, color='r', label='forecast')
        ax.legend()
        ax.tick_params(axis = 'x', rotation = 85)
        ax.set_title(f'1-step ahead Price Return Forecast {ticker}', fontsize=18)
        ax.set_ylabel('Price Return', fontsize=16)

        # save figure to PDF
        pp.savefig(fig)

    pp.close()

if __name__ == '__main__' :

    print("Training TCN Model for every Ticker")

    main()
