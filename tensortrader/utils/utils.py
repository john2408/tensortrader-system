import pandas as pd
import numpy as np
import datetime


SYMBOLS = ['BNBUSDT', 'BNBBTC', 'BTCUSDT', 'EOSUSDT', 'ETCUSDT',
       'LTCUSDT', 'XMRBTC', 'TRXUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT',
       'MKRUSDT', 'DOGEUSDT','ETHUSDT']

def format_cols_to(df, cols, dtype = 'float64'):
    """Format a list of columns to desired
    datatype.

    Args:
        df (pd.DataFrame): input dataframe
        cols (list): columns list
        dtype (str, optional): Data type. Defaults to 'float64'.

    Returns:
        _type_: _description_
    """

    for col in cols:
            df.loc[:,col] = df[col].astype(dtype)

    return df

def reindex_by_date(df):
    """
    Reindex Time Series. 
    """
    dates = range(df.index[0], df.index[-1]+60000,60000)
    return df.reindex(dates, method = 'pad')


def todatetime(t):
    """
    Convert Unix timestamp to datetime.
    """
    return datetime.fromtimestamp(t/1000)