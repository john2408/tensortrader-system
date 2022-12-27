import pandas as pd
import numpy as np
from binance.client import Client
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
import time
import json
import logging

import warnings
warnings.filterwarnings("ignore")

class BinanceTrader():
    
    def __init__(self, 
                symbol : str, 
                signal_loc : str,
                bar_length : str, 
                client : Client,
                model : str, 
                units : float, 
                max_trades : int,
                max_trade_time : int,
                logger : logging.Logger,
                position : int = 0
                ):
        
        self.symbol = symbol
        self.signal_loc = signal_loc        
        self.client = client
        self.bar_length = bar_length
        self.units = units
        self.max_trades = max_trades
        self.max_trade_time = max_trade_time
        self.position = position
        self.model = model
        self.logger = logger
    
        self.trades = 0 
        self.trade_values = []
        self.signal = None
        self.signal_time = None
        self.cum_profits = 0
    

    def new_signal(self) -> bool:
        """Read new signal from database

        Returns:
            bool: whether new trading signal is available
        """
        
        new_signal_time = self.df_signal['creation_time'].iloc[0]
        new_signal = self.df_signal['position'].values[0]
        
        print("Latest signal time:  {}".format(self.signal_time))
        print("New signal time : {}".format(new_signal_time))
        
        if (new_signal_time != self.signal_time):
                    
            self.signal_time = new_signal_time
            self.signal = new_signal

            self.logger.info("New Signal Available, Signal is {}".format(self.signal))
            print("New Signal is: ", self.signal)
            return True
        
        self.logger.info("NO New Signal Available yet")
        print("NO New Signal Available yet")
        return False
        
    
    def get_signal_data(self) -> None:
        """Reading new signal data from database.
        """
        
        df_signal = pd.read_parquet(signal_loc)
            
        df_signal = df_signal[df_signal['ticker'] == self.symbol].copy()
        
        # Get latest signal data
        self.df_signal = df_signal[df_signal['creation_time'] == df_signal['creation_time'].max()].copy()

        self.logger.info("Reading Signal data from database")
        
    def manage_position(self) -> bool:
        """Define Rule to stop/hold trades. 
        
        Target trails. 
        Stop trails.
        
        Returns:
            bool: True: Continue trading. False: Stop trading execution.
        """
        
        # (1) Max Number of trades per session
        # stop trading session
        if self.trades >= self.max_trades: # stop stream after 10 trades per day
            
            self.logger.info("More than {} Trades executed. Stoping traiding execution.".format(self.max_trades))
            if self.position == 1:
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            elif self.position == -1:
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = -1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            else: 
                self.logger.info("STAYING NEUTRAL")
                print("STAYING NEUTRAL")
            
            return True
        
        # (2) Max Holding Time
        info = "Trading Start Time was : {} ".format(self.trade_time)
        print(info)
        self.logger.info(info)
        
        current_trading_time = datetime.utcnow() - pd.to_datetime(self.trade_time)
        max_trading_time = timedelta(minutes = self.max_trade_time)
        
        info = "Current trading holding time {}".format(current_trading_time)
        print(info)        
        self.logger.info(info)
        
        if current_trading_time >= max_trading_time:
            
            info = "More than max time time allowed {}. Closing Trade".format(self.max_trade_time)
            self.logger.info(info)
            print(info)
            
            if self.position == 1:
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            elif self.position == -1:
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = -1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            else: 
                self.logger.info("STAYING NEUTRAL")
                print("STAYING NEUTRAL")
            
            return False
            
        return False
   
    def execute_trades(self): 
        
        
        if self.signal == 1: # if signal is long -> go/stay long
            
            if self.position == -1:
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING CLOSING SHORT TRADE") 
                print("GOING CLOSING SHORT TRADE")
                
                time.sleep(0.1)
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG") 
                print("GOING LONG")
                  

            elif self.position == 1:
                
                self.logger.info("STAYING LONG") 
                print("STAYING LONG")
                
            
            elif self.position == 0:
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                                 
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG")
                print("GOING LONG")
            
            self.position = 1  
                 
        elif self.signal == -1: # if signal is short -> go/stay long
            
            if self.position == 0:
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
                
            elif self.position == 1:
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
                time.sleep(0.1)
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT")
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
            
            elif self.position == -1:
                
                self.logger.info("STAYING SHORT")
                print("STAYING SHORT")
                
                
            self.position = -1

        elif self.signal == 0: # if signal is neutral -> close any trade
            
            if self.position == 1:
                
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
            elif self.position == -1:
                
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
            self.position = 0
        
        
    def report_trade(self, order, going): 
        
        # extract data from order object
        side = order["side"]
        self.trade_time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        # print trade report
        info_trading_side = "{} | {}".format(self.trade_time, going)
        info_transaction_details = "{} | Base_Units = {} | Quote_Units = $ {} | Price = $ {} ".format(self.trade_time, base_units, quote_units, price)
        info_profit_details = "{} | Profit = {} | CumProfits = {} ".format(self.trade_time, real_profit, self.cum_profits)
        
        print(100 * "-" + "\n")
        for info in [info_trading_side, info_transaction_details, info_profit_details]:
            print(info) 
            self.logger.info(info)
        print(100 * "-" + "\n")
        
        
    def start_trading(self) -> None:
        """Main method to execute trading for a given symbol. 
        """
        
        while True:
            
            info = "\nTRADING LOG for {} | Binance Test Net".format(self.symbol)
        
            print(info)     
            self.logger.info(info)
            
            # Get Latest Trading Signals
            self.get_signal_data()
            
            info = "\nTRADING LOG for {} | Position is {}".format(self.symbol, self.position)
            print(info)
            self.logger.info(info)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            
            if self.new_signal():
                
                info = "{} New Signal available".format(timestamp)
                print(info)
                self.logger.info(info)
                
                self.execute_trades()
                info = "{} .".format(timestamp)
                print(info)
                self.logger.info(info)
                
            else:
                
                info = "{} Managing Position".format(timestamp)
                print(info)
                self.logger.info(info)
                
                stop = self.manage_position()
                
                if stop: 
                    print("Finishing Trading")
                    break                    
                
                info = "{} Staying in Position".format(timestamp)
                print(info,)
                self.logger.info(info)
            
            time.sleep(30)

if __name__ == "__main__":
    
    
    import os 
    from tensortrader.tasks.task_utils import create_logging
    
    # -----------------------------
    # Parameters
    # -----------------------------
    symbol = "BTCUSDT"
    units = 0.01
    max_trades = 10
    signal_loc = '/mnt/c/Tensor/Database/TRADING_SIGNALS/Trading_Signals.parquet'
    config_loc = "/mnt/d/Tensor/tensortrader-system/config.json"
    path_logs = "/mnt/d/Tensor/tensortrader-system/logs/trading_execution_logs"
    model = 'TCN' 
    bar_length = "15m"
    position = 0
    max_trade_time = 30 # minutes
    
    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")  
    trading_dir = os.path.join( path_logs,  f"Trading_Execution_LOG_{symbol}")

    if not os.path.exists(trading_dir):
        os.mkdir(trading_dir)

    print("Storing Price Return data at", trading_dir)

    LOG_FILENAME = os.path.join( trading_dir, f"{timestamp}_Trading_Execution.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)
    
    # -----------------------------
    # Trader
    # -----------------------------   
    with open(config_loc) as f:
        CONF = json.load(f)
    f.close()
    
    api_key = CONF.get('key_test')
    secret_key = CONF.get('secret_test')

    try:
        client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)
        logger.info("Connection to Binance Test API sucessfully created.")
    except Exception as e:
        logger.error(f"{e}")
    
    
    trader = BinanceTrader(symbol = symbol,
                signal_loc = signal_loc, 
                bar_length = bar_length,  
                client  = client, 
                model = model,
                units = units, 
                position = position, 
                max_trades = max_trades,
                max_trade_time = max_trade_time,
                logger = logger)
    
    trader.start_trading()