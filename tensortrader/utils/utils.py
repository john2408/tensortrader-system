import pandas as pd
import numpy as np

import pytz
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

def utc_to_local(utc_dt : datetime.datetime, local_tz: pytz.tzfile ) -> datetime.datetime:
    """Convert UTC datetime to local time

    Args:
        utc_dt (datetime): datetime date
        local_tz (pytz.tzfile): local time zone object

    Returns:
        datetime: transformed datetime object
    """
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_dt