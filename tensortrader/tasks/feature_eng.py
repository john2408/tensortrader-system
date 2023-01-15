import os
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml

#from tensortrader.Features import feature_func as fe
from tensortrader.Features import Feature as fe

if __name__ == '__main__':

    # -------------------------------------------------------
    # (1) Param Config
    # -------------------------------------------------------
    storage_folder = '/mnt/c/Tensor/Database/Cryptos/TBM/'
    CONF = yaml.safe_load(Path('../config.yml').read_text())

    #storage_folder = '/mnt/Data/Tensor_Invest_Fund/data/Cryptos/TBM/'

    # Feature Configuration ID
    feature_id = 1

    # Strategy
    strategy = '1-1_vb_15m' # '2-1_vb_15m' or 2-2_vb_15m

    SYMBOLS = ['ADAUSDT',
    'BNBBTC',
    'BNBUSDT',
    'BTCUSDT',
    'DOGEUSDT',
    'EOSUSDT',
    'ETCUSDT',
    'ETHUSDT',
    'IOTAUSDT',
    'LTCUSDT',
    'MKRUSDT',
    'TRXUSDT',
    'XLMUSDT',
    'XMRBTC']

    SYMBOLS = ['ADAUSDT',
            'BNBBTC',
            'BNBUSDT',
            'ETHUSDT']

    # -------------------------------------------------------
    # (2) Data Load
    # -------------------------------------------------------
    input_folder_db = storage_folder

    dfs = []

    for ticker in SYMBOLS:

        data_path = f'{ticker}/Tripe_Barrier_Method_{ticker}_ptsl_{strategy}.parquet'

        dfs.append(pd.read_parquet(os.path.join(input_folder_db, data_path)))

    data = pd.concat(dfs, ignore_index= True)

    # -------------------------------------------------------
    # (3) Features Calculation
    # -------------------------------------------------------

    # Features Configuration
    features_conf = CONF['Feature_Engineering'][feature_id]

    # (1) Calculate Technical Indicators
    ta_config = features_conf['ta']
    data = fe.calculate_technical_indicators(data, features_conf, SYMBOLS)

    # (2) Calculate Lag Features
    if features_conf['include_lags']:
        data = fe.calculate_lag_features(data, features_conf, SYMBOLS)

    # (3) Calculate Return Features
    if features_conf['Return_Features']:
        date_col = 'Date'
        data = fe.calculate_returns_per_ticker(data, features_conf, SYMBOLS, date_col)

    # (4) Momemtum Features
    if features_conf['Return_Features'] and features_conf['Momentum_Features']:
        data = fe.calculate_momemtum_features(data, features_conf, SYMBOLS)

    # (5) Time Features
    if features_conf['Time_Features']:

        time_levels =  ['month', 'day', 'hour', 'minute']
        timestamp_col = 'Date'
        data = fe.build_time_columns(data, timestamp_col, time_levels)

        if features_conf['Time_Fourier_Features']:
            data = fe.build_fourier_time_features(data, time_levels = ['month', 'day', 'hour', 'minute'], max_levels = [12, 30, 24, 60], drop_columns = True)

    # (6) Volume Features
    if features_conf['Volume_Features']:
        group_level = ['Ticker']
        data = fe.calculate_volume_features(data, group_level, features_conf)

    # (7) Apply Standard Scaler
    if features_conf['Apply_Standard_Scaler']:

        if features_conf['Apply_Standard_Scaler_Lags']:

            cols_to_add = []
            for lag_variable in features_conf['ref_variable_lags']:
                for lag in features_conf['lags']:
                    cols_to_add.append(f'{lag_variable}_lag_{lag}')


            cols = features_conf['Standard_Scaler_Cols'] + cols_to_add
        else:
            cols = features_conf['Standard_Scaler_Cols']

        print(cols)

        for col in cols:
            data.loc[:,f'{col}_standard'] = data.groupby('Ticker')[col].transform(lambda x: fe.apply_standard_scaler(x))


    # -------------------------------------------------------
    # (8) Metalabels
    # -------------------------------------------------------
    windows = [5,30]
    group_level = ['Ticker']

    for window in windows:

        data[f'SMA_{window}'] = data.groupby(group_level)['Close'].transform(lambda x: x.rolling(window = window, closed = 'left').mean())

    data = fe.strategy_crossing_sma(data, sma_w = windows)

    data.loc[:,'metalabel'] = fe.get_metalabels(y_model1 = data['sma_cross_over'] , y_true = data['label'])

    print(data['metalabel'].value_counts())

    # -------------------------------------------------------
    # (9) Data Storage
    # -------------------------------------------------------
    output_folder_db ='/media/john/Data/Tensor_Invest_Fund/data/Cryptos/Features_Eng'

    sub_experiment_type = 'conf_{}_Tickers_{}_Stategy_{}'.format(feature_id, data['Ticker'].nunique(), strategy)

    output_location = os.path.join(output_folder_db,
                                ('Feature_Engineering_{}.parquet'
                                .format( sub_experiment_type )))

    print(output_location)

    data.to_parquet(output_location , engine = 'fastparquet', compression = 'gzip')
