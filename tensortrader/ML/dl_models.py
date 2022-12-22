import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras as ks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN


import yaml
from pathlib import Path
import joblib

def test_train_batches(lag_length: int, 
                      ts_train : np.ndarray, 
                      ts_test: np.ndarray) -> list:
    """Get Test/Train batches for as Tensors

    Args:
        lag_length (int): _description_
        ts_train (np.ndarray): _description_
        ts_test (np.ndarray): _description_

    Returns:
        list[np.ndarray]: _description_
    """
    # 1-step-ahead forecast
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for i in range(lag_length, len(ts_train)):
        X_train.append(ts_train[(i - lag_length):i])
        Y_train.append(ts_train[i])
        
    for i in range(lag_length, len(ts_test)): 
        X_test.append(ts_test[(i - lag_length):i])
        Y_test.append(ts_test[i])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print(X_train.shape)
    print(Y_train.shape)
    print()
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def train_tcn_model(X_train : np.ndarray, 
                Y_train : np.ndarray, 
                X_test : np.ndarray, 
                Y_test: np.ndarray, 
                lag_length : int, 
                epochs: int, 
                verbose : int, 
                n_features: int, 
                seed : int, 
                dilations : list, 
                kernel_size : int):

    ks.utils.set_random_seed(123)
    model = Sequential()
    model.add(TCN(input_shape=(lag_length, n_features),
            dilations = dilations,
            kernel_size = kernel_size,
            use_skip_connections=True,
            use_batch_norm=False,
            use_weight_norm=False,
            use_layer_norm=False
            ))
    model.add(Dense(2, activation='linear')) 
    model.add(Dense(1, activation='linear')) # https://keras.io/api/layers/activations/
    model.compile(optimizer='adam', loss='mse')

    # Define Early stopping
    early_stop = EarlyStopping(monitor='val_loss',patience=5)

    print('Train...')
    # fit model
    model.fit(X_train, Y_train,
                epochs=epochs,
                validation_data = (X_test, Y_test),
                callbacks=[early_stop], 
                verbose = verbose)
                
    model.summary()

    return model