dfs_train = []
dfs_test = []
dfs_metric = []

for ticker in df_prices['ticker'].unique():
    
    print("Training model for ticker ", ticker)

    df_temp = df_prices[df_prices['ticker'] == ticker].copy()

    ts = df_temp['denoised_price_returns'].values.reshape(-1, 1)
    
    # Take the max divisible units of the timeseries
    # which can be generated by the selected batch_size
    lag_length = df_temp['pacf_lag'].values[0]
    max_len = int(len(ts)/lag_length) *lag_length
    ts = ts[-max_len:]

    scaler = StandardScaler()
    ts_norm = scaler.fit_transform(ts)

    test_units = int(len(ts) * test_size)
    test_start = int(len(ts) - test_units)

    ts_train = ts_norm[:test_start]
    ts_test = ts_norm[test_start:]

    X_train, Y_train, X_test, Y_test = test_train_batches(lag_length =lag_length, 
                                    ts_train = ts_train, 
                                    ts_test = ts_test)

    model = train_tcn_model(X_train = X_train, 
                Y_train = Y_train, 
                X_test = X_test, 
                Y_test = Y_test,
                lag_length = lag_length, 
                epochs = epochs, 
                verbose = verbose, 
                n_features = n_features, 
                seed = seed, 
                dilations = dilations, 
                kernel_size = kernel_size)               

    forecast_train = scaler.inverse_transform(model.predict(X_train)).reshape(1,-1)[0]
    forecast_test = scaler.inverse_transform(model.predict(X_test)).reshape(1,-1)[0]

    original_train = scaler.inverse_transform(Y_train).reshape(1,-1)[0]
    original_test = scaler.inverse_transform(Y_test).reshape(1,-1)[0]

    df_train = pd.DataFrame()
    df_train['forecast_train'] = forecast_train 
    df_train['original_train'] = original_train 
    df_train['ticker'] = ticker

    df_test = pd.DataFrame()
    df_test['forecast_test'] = forecast_test 
    df_test['original_test'] = original_test 
    df_test['ticker'] = ticker

    dfs_train.append(df_train)
    dfs_test.append(df_test)

    pearson_coff = np.corrcoef(forecast_test, original_test)[0,1]
    pearson_coff = np.round(pearson_coff, 3)
    print(" Peasron coefficient is: " , np.round(pearson_coff, 3), "for one step ahead forecast" )

    df_metric = pd.DataFrame({'ticker' : [ticker],
                             'pearson_coff': [pearson_coff] })

    dfs_metric.append(df_metric)

    del df_test
    del df_train
    
    # Store File
    print("Storing Model")
    filepath = f'../data/models/TCN_Model_{ticker}'
    model.save(filepath)  

    # Store Scaler
    print("Storing Scaler")
    filepath = f'../data/scalers/Scaler_{ticker}.pkl'
    joblib.dump(scaler, filepath) 