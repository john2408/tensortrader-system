import pandas as pd 
import numpy as np  
import json 
import os
from pathlib import Path
from os.path import join
import dateutil

# https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c
# export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
# export PYTHONPATH="${PYTHONPATH}:/home/john/Projects/Tensor/tensortrader/tensortrader
from tensortrader.utils.utils import *


from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from datetime import date, datetime, timedelta
import time
from collections import OrderedDict

# Binance REST API Guideline
# https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md


class ETL():

    def __init__(self, load_size_days, 
                    end_timestamp, 
                    start_timestamp = None, 
                    total_days = None ):
        """ETL Metaclass

        Args:
            load_size_days (int): number of days to load per request
            end_timestamp (datetime.datetime): end update timestamp
                normally set to datetime.utcnow()
            start_timestamp (datetime.datetime, optional): start update timestamp. 
                    Optional, if not set, then when updating a file lastet timestamp will 
                    be used as start timestamp. 
            total_days (int, optional): Total days to download, when 
                retrieving historical data for a given asset. Defaults to None.
        """

        self.load_size_days = load_size_days
        self.end_timestamp = end_timestamp
        self.start_timestamp = start_timestamp
        self.total_days = total_days
        self._import_timestamps = []

    @property
    def import_timestamps(self):
        return self._import_timestamps

    
    def create_import_timestamps_tuples(self):
        """Create import timestamps for a given 
        stock or cryptocurrency, dividing the 
        time in the number of days per request
        given by the variable 'load_size_days' 
        between start and end timestamp.
        
        Returns:
            list: start and ending timestamps for data retrieval
        """
        
        self._import_timestamps = []

        if self.total_days is not None:

            print("for")
            
            for lot in range(1, int(self.total_days/self.load_size_days)):
                self._import_timestamps.append((self.end_timestamp - timedelta(days = lot * self.load_size_days), 
                                            self.end_timestamp - timedelta(days = (lot -1) * self.load_size_days)))
                print("adfadsfölkj")
        else:
            
            
            # To create daily updates for appending to historical data
            lot = 1
            end_time = self.start_timestamp
            while(end_time < self.end_timestamp):

                start_time = self.start_timestamp + timedelta(days = (lot -1)  * self.load_size_days)
                end_time =  self.start_timestamp + timedelta(days = lot * self.load_size_days)
                
                if end_time > self.end_timestamp:
                    end_time = self.end_timestamp

                self._import_timestamps.append((start_time, 
                                                end_time))

                lot += 1

            
        return self._import_timestamps

    def retrieve_historical_data(self):
        raise NotImplementedError
        

    def update_data(lastest_update_date ):
        raise NotImplementedError



class ETL_Binance(ETL):
    """ETL Class to retrieve data
    using Binance API via python 
    Binance package. 

    Args:
        ETL (class): ETL Metaclass
    """

    def __init__(self, 
                    tickers, 
                    storage_folder,
                    load_size_days, 
                    end_timestamp,
                    start_timestamp = None, 
                    total_days = None):
        """_summary_

        Args:
            tickers (list): coinpairs tickers
            storage_folder (str): storage folder to store data
            load_size_days (int): number of days to load per request
            end_timestamp (datetime.datetime): end update timestamp
                normally set to datetime.utcnow()
            start_timestamp (datetime.datetime, optional): start update timestamp. 
                    Optional, if not set, then when updating a file lastet timestamp will 
                    be used as start timestamp. 
            total_days (int, optional): Total days to download, when 
                retrieving historical data for a given asset. Defaults to None.
        """

        super().__init__(load_size_days,
                            end_timestamp,
                            start_timestamp,
                            total_days  )

        self._tickers = tickers
        self.storage_folder = storage_folder
        self._client = None


    @property     
    def tickers(self):
        return self._tickers

    @property     
    def client(self):
        return self._client

    def connect_API(self, config):
        """Connect to Binance API
        given a key and secret token


        Args:
            config (dict): dictionary containing key and secret
        """
        try:
            self._client = Client(config['key'], config['secret'])
            print("Sucessful connection to Binance API")
        except Exception as e:
            print("Unsucessfull connection, ", e)

    def download_historical_data(self, interval, limit = 1000, verbose = 0):
        """Method to download historical data. 

        Args:
            interval (str): time interval for candle bars
            limit (int, optional): Candle bar limit. Defaults to 1000.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """

        # Calculate Import time stamps
        self._import_timestamps = self.create_import_timestamps_tuples()

        print(self._import_timestamps)

        _df = pd.DataFrame()
        for ticker in self._tickers:

            print(ticker)

            if verbose > 0:
                print("Getting data for ticker", ticker)

            _df = self.retrieve_historical_data(ticker, interval, limit )

            if not _df.empty:
                self.store_historical_as_parquet(_df, ticker)

    
    def update_data(self, interval, limit = 1000, verbose = 0):
        """Update data for existing historical data
        for coinpairs. It updates the current year's file 
        for every ticker in the tickers list. 

        Args:
            interval (str): time interval for candle bars
            limit (int, optional): Candle bar limit. Defaults to 1000.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """

       
        # TODO: adjust function to be able to update multiple
        # years at once, when year end to year begin
        for ticker in self._tickers:
            
            if verbose > 0:
                print("Updating data for ticker", ticker)

            df_old = self.read_latest_file(ticker)

            # Get latest update timestamp
            self.start_timestamp = df_old['Date'].max() + timedelta(minutes=1)

            print("Latest update at", self.start_timestamp )

            # Calculate Import time stamps
            self._import_timestamps = self.create_import_timestamps_tuples()

            if verbose > 0:
                print(" Updating data between", 
                    self._import_timestamps[0][0], 
                    " and ", self._import_timestamps[-1][1])

            df_new = self.retrieve_historical_data(ticker, interval, limit )

            if not df_new.empty:

                # Concanetate new data
                df_new = pd.concat([df_new, df_old], ignore_index= True).copy()

                self.store_update_as_parquet(df_new, ticker, year = max(df_new['Year'].unique()))

                if verbose > 0:
                    print("Sucessfully data update for", ticker, "New Update at", df_new['Date'].max())
                    print("\n")

            else:

                print("No data found to update")
                


    def retrieve_historical_data(self, ticker, interval, limit = 1000 ):
        """Get historical data from Binance API. 
        Avoiding request overload or account ban. 

        Args:
            ticker (str): coinpair ticker
            interval (str): candle bars interval
            limit (int, optional): candle bars limit. Defaults to 1000.

        Returns:
            pd.DataFrame: historical data for selected ticker
        """


        dfs = []
        df_out = pd.DataFrame()

        number_of_requests = 0

        for start_str, end_str in self._import_timestamps:

            print("Getting data from ", start_str, " to , ", end_str)

            try:                
                bars = self._client.get_historical_klines(symbol = ticker, 
                                                    interval = interval,
                                                    start_str = str(start_str), 
                                                    end_str = str(end_str), 
                                                    limit = limit)
            except Exception as e:

                print(" Data could not be retrieved. ")
                print(" Error", e)
                print(" No more data will be imported. Closing Loop")
                break
                
            
            # TODO: Check if all data is in same timezone
            df = pd.DataFrame(bars)
            df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
            df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                            "Clos Time", "Quote Asset Volume", "Number of Trades",
                            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]

            
            dfs.append(df)

            number_of_requests += 1
            
            sleep_time_sec = np.random.randint(2,4)
            print("Waiting ", sleep_time_sec, " Sec...")

            # After 10 requests wait 60 sec for
            # next request
            if number_of_requests == 10:

                print(" Request number reached 10, waiting for 1 minute")

                sleep_time_sec = 60
                number_of_requests = 0
            
            time.sleep(sleep_time_sec)

        
        if dfs:

            #Concatenate all single results
            df_out = pd.concat(dfs, ignore_index = True)

            # Generate year Column
            df_out.loc[:,'Year'] = df_out['Date'].dt.year

            # Generate Ticker column
            df_out.loc[:,'Ticker'] = ticker

        return df_out

    def store_historical_as_parquet(self, df, ticker):
        """Store data as parquet.
        For every coinpair and every year in the history
        a file in the form 
        YYYY_ticker.parquet will be generated.

        Args:
            df (pd.DataFrame): historical candle bar data
            ticker (str): coinpair ticker
        """

        ticker_dir = os.path.join(self.storage_folder, ticker)

        if not os.path.exists(ticker_dir):
            os.mkdir(ticker_dir)

        # Generate year Column
        df.loc[:,'Year'] = df['Date'].dt.year

        # Generate Ticker column
        df.loc[:,'Ticker'] = ticker

        # Format colums to expected dtypes
        df = self.output_df_format(df)

        for year in df['Year'].unique():

            file_name = f"{year}_{ticker}.parquet"

            file_storage_location = os.path.join(ticker_dir, file_name)

            df[df['Year'] == year].to_parquet(file_storage_location)

            print("File Store at ", file_storage_location)

    def store_update_as_parquet(self, df, ticker, year):
        """Store updated data for coinpair.
        An available file of the form YYYY_ticker.parquet
        will be updated with the most recent data. 

        Args:
            df (pd.DataFrame): dataframe with newest data
            ticker (str): ticker
            year (int): year
        """

        ticker_dir = os.path.join(self.storage_folder, ticker)
        file_name = f"{year}_{ticker}.parquet"
        file_storage_location = os.path.join(ticker_dir, file_name)

        df = self.output_df_format(df)

        df.to_parquet(file_storage_location)

        print("File Store at ", file_storage_location)

        pass

    def select_latest_ticker_file(self, ticker):
        """Get storage location for most recent file
        given a ticker

        Args:
            ticker (str): coinpair ticker

        Returns:
            str: storage location for most recent file
        """

        ticker_dir = os.path.join(self.storage_folder, ticker)

        latest_update_year = max([x[:4] for x in os.listdir(ticker_dir)])

        lastest_file = os.path.join(self.storage_folder, 
                                ticker, 
                                f"{latest_update_year}_{ticker}.parquet")

        return lastest_file
    
    def output_df_format(self, df):

        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
       'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
       'Taker Buy Quote Asset Volume']
        df = format_cols_to(df, cols, dtype = 'float64')

        cols = ['Number of Trades', 'Year']
        df = format_cols_to(df, cols, dtype = 'int32')

        return df



    def read_latest_file(self, ticker):
        """Get most recent data for a ticker

        Args:
            ticker (str): coinpair ticker

        Returns:
            pd.DataFrame: data frame with historical data
        """

        lastest_file  = self.select_latest_ticker_file(ticker)

        _df = pd.read_parquet(lastest_file)

        return _df

    def retrieve_all_tickers(self):
        """Retrieve all tickers available in 
        Binance

        Returns:
            list: list with all coinpairs tickers
        """
        
        prices = self.client.get_all_tickers()

        return [x['symbol'] for x in prices]


class DataLoader():
    """Data loader class 

    It loads historical candle basrs data 
    from a parquet datalake for one 
    or multiple tickers. 

    It considers the end date as the current date 
    and calculates the start date using the n_days
    parameter. 
    """

    def __init__(self, input_folder_db: str,) -> None:
        """
        Args:
            input_folder_db (str): database folder
        """
        self.input_folder_db = input_folder_db
        
        self.current_timestamp = None
        self.initial_date_load = None
        self.years_filter = None
     
    def get_years_filter(self) -> None:
        """Get year to load for individual
        database yearly files. 
        """
        start_year = self.initial_date_load.year
        end_year = self.current_timestamp.year
        
        if start_year != end_year:
            self.years_filter = [x for x in range(start_year, end_year + 1)]
        else:
            self.years_filter = [start_year]
        
        

    def load(self, n_days, symbols) -> pd.DataFrame:

        self.current_timestamp = datetime.today()
        self.initial_date_load = self.current_timestamp - dateutil.relativedelta.relativedelta(days = n_days)
        
        self.get_years_filter()
                         
        dfs = []

        for symbol in symbols:
            for year in self.years_filter:

                file_name = os.path.join(self.input_folder_db, 
                                    f'{symbol}', 
                                    f'{year}_{symbol}.parquet')

                print("Reading file", file_name)

                df = pd.read_parquet(file_name)

                print(" Max Date is ", df['Date'].max())

                df = df[df['Date'] >= self.initial_date_load].copy()
                dfs.append(df)

                del df

        data = pd.concat(dfs, ignore_index = True).drop_duplicates()

        data.loc[:,'timestamp'] = data['Open Time'].copy()
        data.set_index('timestamp', inplace = True)
        data.sort_values(by = ['timestamp'], inplace = True)

        data = (data.groupby(['Ticker'], 
                        group_keys= False)
                        .apply(reindex_by_date)
                        .reset_index())

        # Convert Unix timestamp to datetime 
        data.loc[:,'timestamp'] = pd.to_datetime(data['timestamp'], unit = "ms") 

        data = data.drop(columns = ['max_trades'])

        print("Dataframe size is ", data.shape)

        return data
    
    def resampling(self, 
                    df : pd.DataFrame,
                    resampling_value : str,
                    ) -> pd.DataFrame:
        """Resample data frame 
        of the form OHLCVNT
        -> Open, High, Low, Close, Volume, 
        Number of Trades, Ticker

        Args:
            df (pd.DataFrame): input dataframe
            resampling_value (str): resampling string

        Returns:
            pd.DataFrame: _description_
        """

        cols_to_keep = ['timestamp', 'Open', 'High', \
                        'Low', 'Close', 'Volume',  \
                        'Number of Trades','Ticker']

        df = df.filter(cols_to_keep).copy()

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        df = df.set_index('timestamp')

        df = df.groupby('Ticker').resample(resampling_value).agg(
                OrderedDict([
                    ('Open', 'first'),
                    ('High', 'max'),
                    ('Low', 'min'),
                    ('Close', 'last'),
                    ('Volume', 'sum'),
                    ('Number of Trades', 'sum'),
                ])
            )

        data = df.reset_index().copy()
        data['Date'] = data['timestamp'].copy()

        return data

