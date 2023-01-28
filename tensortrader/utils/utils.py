import os
from datetime import datetime

import pandas as pd
import pytz


def format_cols_to(df, cols, dtype="float64"):
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
        df.loc[:, col] = df[col].astype(dtype)

    return df


def reindex_by_date(df):
    """
    Reindex Time Series.
    """
    dates = range(df.index[0], df.index[-1] + 60000, 60000)
    return df.reindex(dates, method="pad")


def todatetime(t):
    """
    Convert Unix timestamp to datetime.
    """
    import datetime

    return datetime.fromtimestamp(t / 1000)


def utc_to_local(utc_dt: datetime, local_tz: pytz.tzfile) -> datetime:
    """Convert UTC datetime to local time

    Args:
        utc_dt (datetime): datetime date
        local_tz (pytz.tzfile): local time zone object

    Returns:
        datetime: transformed datetime object
    """
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_dt


def get_latest_available_folder(folder_loc: str, key_word: str) -> str:
    """Get most recent folder given a timestamp

    Args:
        folder_loc (str): folder location
        key_word (str): keyword

    Returns:
        str: most recent folder location
    """

    folders = [x for x in os.listdir(folder_loc) if key_word in x]

    df_tmp = pd.DataFrame()
    df_tmp["folders"] = folders
    df_tmp["timestamp"] = pd.Series(folders).apply(lambda x: x[:16])
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"], format="%Y-%m-%d-%H-%M")

    # Sort where first row is most recent folder
    df_tmp.sort_values(by=["timestamp"], ascending=False, inplace=True)

    return os.path.join(folder_loc, df_tmp["folders"].values[0])
