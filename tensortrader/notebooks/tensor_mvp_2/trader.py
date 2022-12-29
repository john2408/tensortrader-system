import pandas as pd
import numpy as np
from binance.client import Client
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
import time
import pickle
from xgboost import XGBClassifier
# import plotly.graph_objects as go
# from datetime import datetime
# import pyfolio as pf

import warnings
warnings.filterwarnings("ignore")

class MLTrader():
    
    def __init__(self, symbol, bar_length, model, units, position = 0):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.trades = 0 
        self.trade_values = []
        
        #*****************add strategy-specific attributes here******************
        self.model = model
        #************************************************************************
    
    def start_trading(self, historical_days):
        
        self.twm = ThreadedWebsocketManager(testnet = True)
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, 
                                 interval = self.bar_length,
                                 days = historical_days)
            
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, 
                                        interval = self.bar_length)
        # "else" to be added later in the course 
    
    def get_most_recent(self, symbol, interval, days):
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))
    
        bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000)
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        # stop trading session
        if self.trades >= 5: # stop stream after 5 trades
            self.twm.stop()
            if self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 1
            elif self.position == -1:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = -1
            else: 
                print("STOP")
    
        # print out
        print(".", end = "", flush = True) # just print something to get a feedback (everything OK) 
    
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades()
        
    def define_strategy(self): # "strategy-specific"
        
        df = self.data.copy()
        
        #******************** define your strategy here ************************
        # Features
        df['ADX'] = ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=5)
        df['RSI'] = ta.RSI(df['Close'].values, timeperiod=5)
        df['SMA'] = ta.SMA(df['Close'].values, timeperiod=10)
        
        predictors_list = ['ADX', 'RSI', 'SMA']

        df["position"] = xgb.predict(df[predictors_list])
        #***********************************************************************
        
        self.prepared_data = df.copy()
    
    def execute_trades(self): # NEW!
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == -1:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                print("GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == -1: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                print("GOING NEUTRAL")
            self.position = -1

if __name__ == "__main__":
    
    api_key = ""
    secret_key = ""

    client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)

    twm = ThreadedWebsocketManager()

    symbol = "BTCUSDT"
    bar_length = "1m"
    units = 0.01
    position = 0
    '/Users/abc/Desktop/codedata.pkl'

    xgb = pickle.load(open("Udemy/tensor_mvp_2/xgb_clf.pkl", "rb"))

    trader = MLTrader(symbol = symbol, bar_length = bar_length, model = xgb, units = units, position = position)
    trader.start_trading(historical_days = 3)