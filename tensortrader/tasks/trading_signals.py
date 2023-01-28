import os
import warnings
from pathlib import Path

import joblib
import keras as ks
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

import datetime
from datetime import timedelta

from tensortrader.constants import *
from tensortrader.tasks.task_utils import create_logging
from tensortrader.transformations.denoising import *
from tensortrader.utils.utils import get_latest_available_folder, utc_to_local

# linux: export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system/"
# win: ---


def main():

    # -----------------------------
    # Get config values
    # -----------------------------
    path = "/mnt/d/Tensor/tensortrader-system/tensortrader/config/tcn_training.yml"
    # path = '../config/tcn_training.yml'
    CONF = yaml.safe_load(Path(path).read_text())

    # Candle Stick Upsample Range (in minutes)
    minute_sampling = CONF["minute_sampling"]
    # lookback subset to PACF pattern search
    pacf_days_subset = CONF["pacf_days_subset"]
    # Minute sampling
    minute_sampling = CONF["minute_sampling"]

    # Length of Subset Timeseries
    subset_wavelet_transform = int(
        24 * (60 / minute_sampling) * pacf_days_subset
    )  # 60*hours # Number

    # Input data Loc - historical return prices
    input_data_path_price_return = CONF["input_data_path_price_return"]
    db_name_price_return = CONF["db_name_price_return"]

    # Model Location
    model_storage_loc = CONF["model_storage_loc"]
    output_data_trading_signals = CONF["output_data_trading_signals"]
    key_word = CONF["key_word"]

    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    # path = Path(os.getcwd()).parents[0].parents[0]
    path = "/mnt/d/Tensor/tensortrader-system/logs/"
    signal_log_dir = os.path.join(
        path, "trading_signal_logs", f"Trading_signal_{timestamp}"
    )

    if not os.path.exists(signal_log_dir):
        os.mkdir(signal_log_dir)

    with open(os.path.join(signal_log_dir, "config.yml"), "w") as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Trading singal log at", signal_log_dir)

    LOG_FILENAME = os.path.join(signal_log_dir, f"{timestamp}_Trading_Signal.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Data Load
    # -----------------------------

    logger.info("Getting most recent trained ML Model")
    model_latest_loc = get_latest_available_folder(
        folder_loc=model_storage_loc, key_word=key_word
    )

    print(f"Most Recent trained ML Model available at {model_latest_loc}")
    logger.info(f"Most Recent trained ML Model available at {model_latest_loc}")

    # Temporal Convolutatin Networks Params
    n_features = CONF["n_features"]

    # Denoising Method Parameters
    denoising_method = CONF["denoising_method"]
    thresh = CONF["thresh"]
    wavelet = CONF["wavelet"]

    # Latest Prices returns
    try:
        logger.info("Reading latest price returns")
        path_loc = os.path.join(input_data_path_price_return, db_name_price_return)
        df = pd.read_parquet(path_loc)
        df.sort_values(by=["ticker", "Date"], inplace=True)
    except Exception as e:
        print(f"Erorr {e}")
        logger.info(f"{e}")

    # PACF Lags per ticker
    try:
        logger.info("Reading PACF lags used in training")
        filepath = os.path.join(model_latest_loc, f"PACF_lags.pkl")
        ticker_pacf_lags = joblib.load(filepath)
    except Exception as e:
        print(f"Erorr {e}")
        logger.info(f"{e}")

    # -----------------------------
    # Generate Signal
    # -----------------------------
    df_signals = pd.DataFrame()
    df_signals["ticker"] = SYMBOLS
    df_signals["Signal"] = "NEUTRAL"

    for ticker in df_signals["ticker"].unique():

        print("Getting Predicition for ticker", ticker)
        logger.info(f"Getting Predicition for ticker {ticker}")

        try:
            # load NN Model
            filepath = os.path.join(model_latest_loc, f"TCN_Model_{ticker}")
            print("Reading model at loc", filepath)
            model = ks.models.load_model(filepath)
        except Exception as e:
            print(f"Erorr {e}")
            logger.info(f"{e}")

        try:
            # load Scaler
            filepath = os.path.join(model_latest_loc, f"Scaler_{ticker}.pkl")
            print("Reading model at loc", filepath)
            scaler = joblib.load(filepath)
        except Exception as e:
            print(f"Erorr {e}")
            logger.info(f"{e}")

        # Get last prices
        # (1) Database with 15m candle bars
        prices_return = df[df["Ticker"] == ticker]["Close_target_return_15m"].values

        # Get selected subset of data
        # (2) Calculate price return
        prices_return = prices_return[-subset_wavelet_transform:]

        # (3) Remove High Frequency Noise from Timeseries
        denoiser = denoising(signal=prices_return)

        if denoising_method == "wavelet":
            denoised_prices_return = denoiser.wavelet_denoising(
                thresh=thresh, wavelet=wavelet
            )

        # PACF Lag
        # Step 0. From Model Training
        pacf_lag = ticker_pacf_lags[ticker]

        # (4) Get Signal Prediction
        # Get input data for model
        batch = denoised_prices_return[-pacf_lag:]

        print("PACF lag for ticker", ticker, " is: ", pacf_lag)
        logger.info(f"PACF lag for ticker {ticker} is {pacf_lag}")

        # Adjust Tensor Input shape
        batch = batch.reshape(1, pacf_lag, n_features)

        # Return prediction
        model_prediction = model.predict(batch)
        return_prediction = scaler.inverse_transform(model_prediction).reshape(1, -1)[0]

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
            df_signals["Signal"] = np.where(
                df_signals["ticker"] == ticker, "LONG", df_signals["Signal"]
            )
            print("Going Long")
        elif return_prediction < 0:
            df_signals["Signal"] = np.where(
                df_signals["ticker"] == ticker, "SHORT", df_signals["Signal"]
            )
            print("going Short")
        else:
            df_signals["Signal"] = np.where(
                df_signals["ticker"] == ticker, "NEUTRAL", df_signals["Signal"]
            )
            print("Going neutral")

        last_timestamp = df[df["Ticker"] == ticker]["timestamp_local"].iloc[-1]
        df_signals["timestamp_local"] = last_timestamp

        del model
        del scaler

    logger.info("\tAdding Forecasting Timestamp")
    # Since from label_methods.calculate_returns() the values are shifted one position in the past, when generating
    # the signals the reference forecast value will be 2 position in the future
    df_signals["timestamp_signal_ref"] = df_signals["timestamp_local"].apply(
        lambda x: x + timedelta(minutes=minute_sampling * 2)
    )
    df_signals["creation_time"] = datetime.datetime.now()
    df_signals.drop(columns=["timestamp_local"], inplace=True)

    logger.info("\tStoring Signals")
    filepath = os.path.join(output_data_trading_signals, f"Trading_Signals.parquet")

    if not os.path.exists(filepath):
        df_signals["position"] = df_signals["Signal"].map(
            {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}
        )
        df_signals.drop_duplicates().to_parquet(filepath)
    else:
        df_signals_old = pd.read_parquet(filepath)
        df_signals = pd.concat([df_signals, df_signals_old])
        df_signals["position"] = df_signals["Signal"].map(
            {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}
        )

        # Overwrite values
        df_signals.drop_duplicates().to_parquet(filepath)


if __name__ == "__main__":
    print("Generating Trading Signal")
    main()
