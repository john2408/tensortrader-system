import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

def get_metalabels(y_model1, y_true):
    """Calculate Metalabels

    Args:
        y_model1 (pd.Series): Trading Strategy containing
                          1: Long trades
                          0: No trade
                          -1: Short trades

        y_true (pd.Series): Tripe Barrier Method labels

    Returns:
        np.array: metalabels:
          1: Take the trade
          0: ignore the trade
    """
    
    bin_label = np.zeros_like(y_model1)
    for i in range(y_model1.shape[0]):
        if y_model1[i] != 0 and y_model1[i]*y_true[i] > 0:
            bin_label[i] = 1  # true positive

    return bin_label



def apply_standard_scaler(x):
    """Apply Standard scaler on a column. 

    Args:
        x (pd.Series): data

    Returns:
        np.array: stardardized variable
    """
    
    vector_data = x.values.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(vector_data)
    out = scaler.transform(vector_data)
    
    return out.reshape(1,-1)[0]

def strategy_crossing_sma(_df, sma_w):
    """Golden cross occurs when short SMA crosses long SMA from below.
    Death cross is an opposite situation, when short SMA crosses long SMA from above.

    Args:
        _df (_type_): _description_
        sma_w (_type_): _description_
    """

    sma_short = sma_w[0]
    sma_long = sma_w[1]

    
    _df.loc[:,'s_sma_<_l_sma'] = pd.Series(np.where( _df[f'SMA_{sma_short}'] < _df[f'SMA_{sma_long}'], 1 , -1  ))
    _df.loc[:,'s_sma_<_l_sma_lag1'] = pd.Series(np.where( _df[f'SMA_{sma_short}'] < _df[f'SMA_{sma_long}'], 1 , -1  )).shift(1)
    
    _df.loc[:,'cross'] = np.where( _df['s_sma_<_l_sma'] != _df[f's_sma_<_l_sma_lag1'], 1 , 0  )
    _df.loc[:,'sma_cross_over'] = np.where( _df['cross'] == 0, 0, _df['cross'] * _df['s_sma_<_l_sma_lag1'])

    return _df

def calculate_technical_indicators(data: pd.DataFrame, features_conf: dict, SYMBOLS: list):
    """Function to calculate technical indicators

    Args:
        data (pd.DataFrame): data containing ticker information
        features_conf (dict): technical indicators configuration 
        SYMBOLS (list): tickers list

    Returns:
        pd.DataFrame: df containing technical indicators data
    """


    # Ref: https://github.com/twopirllc/pandas-ta/blob/main/examples/PandasTA_Strategy_Examples.ipynb

    dfs = []

    for ticker in SYMBOLS:

        _df = data[data['Ticker'] == ticker].copy()

        print("Calculating Technical Indicators for ticker", ticker)

        MNQ_strategy = ta.Strategy(
            name="MNQ Strategy",
            description="Non Multiprocessing Strategy by rename Columns",
            ta = features_conf['ta']
        )


        #data.set_index(['datetime'], inplace  = True)

        # Run it Technical Indicators Strategy
        _df.ta.strategy(MNQ_strategy)

        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)


def calculate_lag_features(data: pd.DataFrame, features_conf: dict, SYMBOLS: list):
    """Function to calculate lag features

    Args:
        data (pd.DataFrame): data containing ticker information
        features_conf (dict): technical indicators configuration 
        SYMBOLS (list): tickers list

    Returns:
        pd.DataFrame: df containig lag features
    """

    dfs = []

    for ticker in SYMBOLS:

        print("Calculating lags for ticker", ticker)

        _df = data[data['Ticker'] == ticker].copy()

        n_lags = features_conf['lags']
        ref_variable_lags = features_conf['ref_variable_lags']
        drop = features_conf['drop_lags']

        for ref_variable in ref_variable_lags:

            lags_features = []

            if n_lags is not None:
                for lag in n_lags:

                    columns_name = f'{ref_variable}_lag_{lag}'

                    _df.loc[:,columns_name] = _df[ref_variable].shift(lag)

                    lags_features.append(columns_name) 

        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)

    
def calculate_returns(data: pd.DataFrame, variable: str, 
                    lags: list, binary_lags: bool, date_col : str = 'Date', 
                    outlier_cutoff : float = 0.01):
    """Calculate returns base of a target variable. 

    Args:
        data (pd.DataFrame): data containing ticker information
        variable (str): target variable to calculate returns for
        lags (list): list of lags
        binary_lags (bool): whether to convert lags to binary
        date_col (str, optional): Column holding ticker timestamps. Defaults to 'Date'.
        outlier_cutoff (float, optional): returns' outlier cutoff

    Returns:
        pd.DataFrame: Dataframe containing returns features
    """

    returns = []

    for lag in lags:
        if binary_lags:
            _return = returns.append(data.set_index([date_col])[variable]
                        .sort_index() # Sort by Date
                        .pct_change(lag) # Calculate percentage change of the respective lag value
                        .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                upper=x.quantile(1-outlier_cutoff))) # Cutoff outliers
                        .add(1) # add 1 to the returns
                        .pow(1/lag) # apply n root for n = lag
                        .sub(1) #substract 1
                        .apply(lambda x: 1 if x > 0 else 0)
                        .to_frame(f'{variable}_return_{lag}m')
                        
                        )

        else:
            _return = returns.append(data.set_index([date_col])[variable]
                    .sort_index() # Sort by Date
                    .pct_change(lag) # Calculate percentage change of the respective lag value
                    .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                            upper=x.quantile(1-outlier_cutoff))) # Cutoff outliers
                    .add(1) # add 1 to the returns
                    .pow(1/lag) # apply n root for n = lag
                    .sub(1) #substract 1
                    .to_frame(f'{variable}_return_{lag}m')
                    
                )

    returns.append(_return)
        
    returns = pd.concat(returns, axis = 1)
    data = data.set_index([date_col]).join(returns).dropna()
    data.reset_index(inplace = True)

    return data

def calculate_returns_per_ticker(data: pd.DataFrame, features_conf: dict, 
                                SYMBOLS: list, date_col: str = 'Date'):
    """Function to calculate lag features

    Args:
        data (pd.DataFrame): data containing ticker information
        features_conf (dict): technical indicators configuration 
        SYMBOLS (list): tickers list
        date_col (str, optional): Column holding ticker timestamps. Defaults to 'Date'.

    Returns:
        pd.DataFrame: df containig lag features
    """


    dfs = []

    for ticker in SYMBOLS:

        print("Calculating returns for ticker", ticker)

        _df = data[data['Ticker'] == ticker].copy()

        outlier_cutoff = 0.01
        lags = features_conf['return_lags'] 
        binary_lags = features_conf['binary_lags'] 
        variable = features_conf['return_lags_variable'] 

        _df = calculate_returns(_df, variable, lags, binary_lags, date_col, outlier_cutoff)

        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)


def calculate_momemtum_features(data: pd.DataFrame, features_conf: dict, SYMBOLS: list):
    """Function to calculate lag features

    Args:s
        data (pd.DataFrame): data containing ticker information
        features_conf (dict): technical indicators configuration 
        SYMBOLS (list): tickers list

    Returns:
        pd.DataFrame: df containig lag features
    """
    
    dfs = []

    for ticker in SYMBOLS:

        print("Calculating momemtum for ticker", ticker)

        _df = data[data['Ticker'] == ticker].copy()


        lags = features_conf['return_lags'] 
        variable = features_conf['return_lags_variable']

        for lag in lags:
            if lag > lags[0]:
                _df['momentum_{}_{}'.format( lags[0], lag)] = data[f'{variable}_return_{lag}m'].sub(data['{}_return_{}m'.format(variable, lags[0])])
            if lag > lags[1]:
                _df['momentum_{}_{}'.format( lags[1], lag)] = data[f'{variable}_return_{lag}m'].sub(data['{}_return_{}m'.format(variable, lags[1])])

    
        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)


def calculate_volume_features(data: pd.DataFrame, 
                            group_level: list, 
                            features_conf: dict):
    """Function to calculate Volume Features

    Args:s
        data (pd.DataFrame): data containing ticker information
        group_level: (list): grouping level (tickers)
        features_conf (dict): technical indicators configuration

    Returns:
        pd.DataFrame: df containig volume features
    """

    
    short = features_conf['Volume_Windows'][0]
    long = features_conf['Volume_Windows'][1]
    target_variable = features_conf['Volume_Col'] 

    drop_columns = []

    variables = ['sma', 'std']

    data = calculate_rolling_features(data, group_level, 
                                        target_variable, short, long, 
                                        variables, drop_columns, 
                                        drop_target_variable = False )

    
    return data

    

def calculate_rolling_features(df: pd.DataFrame,
                            group_level: list,
                            target_variable: str, 
                            short: int, 
                            long: int, 
                            variables = [], 
                            drop_columns = [],
                            drop_target_variable = True):
    """Function to calculate rolling feature for a target variable in 
    a data frame. 
    
    Args:
        df (pandas.Dataframe): df input data frame
        target_variable (str): df containing the training samples
        short (int): short window
        long (int): long short 
        variables (list): 'sma', 'std', 'bbands' and 'cv'
        drop_columns (list): columns to drop
        drop_target_variable (bool): whether to drop the target variable
    
    Returns:
        pandas.Dataframe: original data frame containing the calculated features
    """

    if 'sma' in variables:
        df[f'{target_variable}_sma_{short}'] = df.groupby(group_level)[target_variable].transform(lambda x: x.rolling(window = short).mean())
        df[f'{target_variable}_sma_{long}'] = df.groupby(group_level)[target_variable].transform(lambda x: x.rolling(window = long).mean())
    
    if 'std' in variables:
        df[f'{target_variable}_std_{short}'] = df.groupby(group_level)[target_variable].transform(lambda x: x.rolling(window = short).std())
        df[f'{target_variable}_std_{long}'] = df.groupby(group_level)[target_variable].transform(lambda x: x.rolling(window = long).std())

    if 'cv' in variables:
        if 'sma' and 'std' in variables:
            df[f'{target_variable}_cv_{short}'] = (df[f'{target_variable}_std_{short}'] 
                                                    / df[f'{target_variable}_sma_{short}'] ) 
            df[f'{target_variable}_cv_{long}'] = (df[f'{target_variable}_std_{long}'] 
                                                    / df[f'{target_variable}_sma_{long}']) 
        else:
            ValueError("Please include sma and std for calculation of cv")
    
    if 'bbands' in variables:
        if 'sma' and 'std' in variables: 
            df[f'{target_variable}_bblow_{short}'] = (df[f'{target_variable}_sma_{short}'] + 1.5 * 
                                                    df[f'{target_variable}_std_{short}'])
            df[f'{target_variable}_bblow_{long}'] = (df[f'{target_variable}_sma_{long}'] + 1.5 * 
                                                    df[f'{target_variable}_std_{long}'])

            df[f'{target_variable}_bbhigh_{short}'] = (df[f'{target_variable}_sma_{short}'] + 2 * 
                                                    df[f'{target_variable}_std_{short}'])
            df[f'{target_variable}_bbhigh_{long}'] = (df[f'{target_variable}_sma_{long}'] + 2 * 
                                                    df[f'{target_variable}_std_{long}'])
        else:
            ValueError("Please include sma and std for calculation of cv")
    
    if drop_columns:
        df.drop(columns = drop_columns, inplace = True)
    
    if drop_target_variable: 
        df.drop(columns = [target_variable], inplace = True)

    
    return df   

def calculate_prob_distribution_features(data: pd.DataFrame, 
                                        target_variable: str,
                                        short : int = 5, 
                                        long : int = 10, 
                                        ):
    """Function daily probability of riksk and target entry
    
    Args:
        
    """

    daily_distribution = (data.groupby(['Date'])[target_variable]
                        .value_counts()
                        .to_frame()
                        .rename(columns = {target_variable: 'counts'})
                        .reset_index())

    daily_distribution['daily_sum'] = daily_distribution.groupby(['Date'])['counts'].transform(np.sum)

    daily_distribution['distribution'] = np.round( daily_distribution['counts'] / daily_distribution['daily_sum'] , 4)

    

    daily_distribution[f'{target_variable}_sma_{short}'] = (daily_distribution.groupby([target_variable])
                                                    ['distribution']
                                                    .transform(lambda x : 
                                                    x.rolling(window = short, closed = 'left')
                                                    .mean().fillna(method = 'backfill')))

    daily_distribution[f'{target_variable}_sma_{long}'] = (daily_distribution.groupby([target_variable])
                                                    ['distribution']
                                                    .transform(lambda x : 
                                                    x.rolling(window = long, closed = 'left')
                                                    .mean().fillna(method = 'backfill')))

    daily_distribution[f'{target_variable}_std_{short}'] = (daily_distribution.groupby([target_variable])
                                                    ['distribution']
                                                    .transform(lambda x : 
                                                    x.rolling(window = short, closed = 'left')
                                                    .std().fillna(method = 'backfill')))

    daily_distribution[f'{target_variable}_std_{long}'] = (daily_distribution.groupby([target_variable])
                                                    ['distribution']
                                                    .transform(lambda x : 
                                                    x.rolling(window = long, closed = 'left')
                                                    .std().fillna(method = 'backfill')))
    # Coefficient of variation
    daily_distribution[f'{target_variable}_cv_{short}'] = (daily_distribution[f'{target_variable}_std_{short}'] 
                                                            / daily_distribution[f'{target_variable}_sma_{short}'])
                                                            
    daily_distribution[f'{target_variable}_cv_{long}'] = (daily_distribution[f'{target_variable}_std_{long}'] 
                                                        / daily_distribution[f'{target_variable}_sma_{long}'])

    if target_variable == 'entry_type':
        daily_distribution = daily_distribution[daily_distribution[target_variable] == 1].copy(deep = True)
    elif target_variable == 'risk_type':
        daily_distribution = daily_distribution[daily_distribution[target_variable] == 0].copy(deep = True)

    return daily_distribution

def build_time_columns(df : pd.DataFrame, 
                        timestamp_col : str = 'Date',
                        time_levels: list =  ['month', 'day', 'hour', 'minute']):
    """_summary_

    Args:
        df (pd.DataFrame): data containing ticker information
        timestamp_col (str, optional): Timestamp column. Defaults to 'Date'.
        time_levels (list, optional): Timelevels. Defaults to ['month', 'day', 'hour', 'minute'].

    Returns:
        pd.DataFrame: df containing time levels columns
    """

    if 'month' in time_levels:
        df.loc[:,'month'] = df[timestamp_col].dt.month

    if 'day' in time_levels:
        df.loc[:,'day'] = df[timestamp_col].dt.day

    if 'hour' in time_levels:
        df.loc[:,'hour'] = df[timestamp_col].dt.hour

    if 'minute' in time_levels:
        df.loc[:,'minute'] = df[timestamp_col].dt.minute

    return df


def build_fourier_time_features(df : pd.DataFrame, 
                                time_levels: list, 
                                max_levels: list, 
                                drop_columns = False):
    """_summary_

    Args:
        df (pd.DataFrame): data containing ticker information
        time_levels (list): list of time levels to transform
        max_levels (list): parameter to calculate number of time units in upper time unit
        drop_columns (bool, optional): Whether to drop the original column. Defaults to False.

    Returns:
        pd.DataFrame: df containing time levels columns transformed by sin and cos
    """

    for time_level, max_level in zip(time_levels, max_levels):
        
        df.loc[:,time_level] = df[time_level].astype('float64')
        
        df.loc[:,f"{time_level}_sin"] = df[time_level].apply(
                                    lambda x: np.sin( 2 * np.pi + x/max_level))
                                    
        df.loc[:,f"{time_level}_cos"] = df[time_level].apply(
                                    lambda x: np.cos( 2 * np.pi + x/max_level))

    if drop_columns: 
        df.drop(columns = time_levels, inplace = True)

    return df 