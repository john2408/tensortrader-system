from pathlib import Path

import joblib
import keras as ks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from tcn import TCN
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class tcn_model():

    def __init__(self,
                    ts_data : np.ndarray,
                    timestamps : pd.Series,
                    test_size : float,
                    lag_length : int,
                    n_features : int,
                    seed : int,
                    dilations : list,
                    kernel_size : int,
                    epochs : int,
                    patience : int = 5,
                    monitor : str = 'val_loss',
                    verbose : int = 0) -> None:
        self.ts_data = ts_data
        self.timestamps = timestamps
        self.test_size = test_size
        self.lag_length = lag_length
        self.n_features = n_features
        self.seed = seed
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        self.monitor = monitor
        self.ts_len = len(ts_data)

        self.scaler = None
        self.model = None
        self.test_train_batches = []
        self.train_test_timestamps = []



    def max_ts_length(self):
        """Recalculate max len of timeseries to be divisible
        by the lag length.

        Take the max divisible units of the timeseries
        which can be generated by the selected batch_size (lag length)
        """


        max_len = int(self.ts_len / self.lag_length) * self.lag_length
        print("Max len for timeseries training", max_len, " from ", self.ts_len )
        self.ts_data = self.ts_data[-max_len:]
        self.timestamps = self.timestamps.iloc[-max_len:]


    def get_test_train_historical_split(self):

        self.scaler = StandardScaler()
        ts_norm = self.scaler.fit_transform(self.ts_data)

        test_units = int(self.ts_len * self.test_size)
        test_start = int(self.ts_len - test_units)

        # Getting Train/Test Tensor np.array
        ts_train, ts_test = ts_norm[:test_start], ts_norm[test_start:]

        # Getting timestamps for train and test dataframes taking into account the lag_length - see: get_test_train_batches()
        start_test_timestamps = test_start + self.lag_length
        self.train_test_timestamps = self.timestamps.iloc[self.lag_length:test_start], self.timestamps.iloc[start_test_timestamps:]

        print("ts_train: ", len(ts_train))
        print("ts_test: ", len(ts_test))

        print("Timestamps train: ", len(self.train_test_timestamps[0]))
        print("Timestamps test: ", len(self.train_test_timestamps[1]))

        return ts_train, ts_test

    def get_test_train_batches(self,
                        ts_train : np.ndarray,
                        ts_test: np.ndarray) -> list:
        """Get Test/Train batches as Tensors

        Args:
            lag_length (int): batch length
            ts_train (np.ndarray): timeseries train
            ts_test (np.ndarray): timeseries test

        Returns:
            list[np.ndarray]: _description_
        """
        # 1-step-ahead forecast
        X_train, Y_train = [], []
        X_test, Y_test = [], []

        for i in range(self.lag_length, len(ts_train)):
            X_train.append(ts_train[(i - self.lag_length):i])
            Y_train.append(ts_train[i])

        for i in range(self.lag_length, len(ts_test)):
            X_test.append(ts_test[(i - self.lag_length):i])
            Y_test.append(ts_test[i])

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        print()
        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print()
        print("X_test shape: ", X_test.shape)
        print("Y_test shape: ", Y_test.shape)

        return X_train, Y_train, X_test, Y_test


    def define_tcn_model_v1(self):

        ks.utils.set_random_seed(self.seed)

        # https://keras.io/api/layers/activations/
        model = Sequential()
        model.add(TCN(input_shape=(self.lag_length, self.n_features),
                dilations = self.dilations,
                kernel_size = self.kernel_size,
                use_skip_connections=True,
                use_batch_norm=False,
                use_weight_norm=False,
                use_layer_norm=False
                ))
        model.add(Dense(2, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        self.model = model

    def fit(self):

        # (1) Calculate max input timeseries lenght for NN
        print("Calculate max ts length")
        self.max_ts_length()

        # (2) Get Train/Test Timeseries Tensor
        print("Getting timeseries train and test")
        ts_train, ts_test = self.get_test_train_historical_split()

        print("Getting train/test batches")
        X_train, Y_train, X_test, Y_test = self.get_test_train_batches(
                                                    ts_train = ts_train,
                                                    ts_test = ts_test)

        # (3) Define a TCN model
        self.define_tcn_model_v1()
        print(self.model.summary())

        # Define Early stopping
        print("Fitting TCN Model")
        early_stop = EarlyStopping(monitor = self.monitor, patience = self.patience)

        # fit model
        self.model.fit(
                    X_train,
                    Y_train,
                    epochs = self.epochs,
                    validation_data = (X_test, Y_test),
                    callbacks = [early_stop],
                    verbose = self.verbose)

        self.test_train_batches = [X_train, Y_train, X_test, Y_test]
        self.model.summary()
