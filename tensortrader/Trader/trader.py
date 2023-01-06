import pandas as pd
import numpy as np
from binance.client import Client
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
import time
import logging
import os
import pymongo
from enum import Enum

import warnings
warnings.filterwarnings("ignore")

class TraderSide(int, Enum):
    SELL = -1
    BUY = 1
    NEUTRAL = 0

# TODO: Implement Stop and Target using Binance API
# TODO: Check stability of https://github.com/binance/binance-connector-python as an option 

class BinanceTrader():
    
    def __init__(self, 
                symbol : str, 
                database_loc : str,
                mongodb_collection : pymongo.collection.Collection,
                signal_loc : str,
                bar_length : str, 
                client : Client,
                model : str, 
                units : float, 
                max_trades : int,
                max_trade_time : int,
                target_usdt : float, 
                stop_usdt : float,
                logger : logging.Logger,
                position : int = 0
                ):
        
        self.symbol = symbol
        self.database_loc = database_loc
        self.mongodb_collection = mongodb_collection
        self.signal_loc = signal_loc        
        self.client = client
        self.bar_length = bar_length
        self.units = units
        self.max_trades = max_trades
        self.max_trade_time = max_trade_time
        self.position = position
        self.model = model
        self.logger = logger
        self.target_usdt = target_usdt
        self.stop_usdt = stop_usdt
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        
        self.trades = 0 
        self.cum_profits = 0
        self.trade_values = []
        self.trade_id = 0
        
        self.signal = None
        self.signal_time = None
        self.current_price = None
        self.event_time = None
        self.entry_price = None
        self.trade_entry_time = None
        self.target_price = None
        self.stop_price = None
        self.trade_profit = None
        self.trading_session_id = None
        self.trade_base_unit = None
        self.trading_data_path = None
        self.trade_orderId = None
        
        # Trade details
        self.trade_data_mongodb = None
        self.trade_data = None
              
        
    def start_streaming(self) -> None:
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            
            os.system('w32tm/resync')  
            
            #https://python-binance.readthedocs.io/en/latest/websockets.html
            self.twm.start_kline_socket(callback = self.handle_socket_message,
                                        symbol = self.symbol, 
                                        interval = self.bar_length)
        
    def handle_socket_message(self, msg) -> None:
        """Handler method using
        the Binance ThreadedWebsocketManager callback.
        
        Update close (current) price and event time.
        
        Args:
            msg (dict): message return in api call
        """
        
        # extract the required items from msg
        event_time  = pd.to_datetime(msg["E"], unit = "ms")
        start_time  = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first       = float(msg["k"]["o"])
        high        = float(msg["k"]["h"])
        low         = float(msg["k"]["l"])
        close       = float(msg["k"]["c"])
        volume      = float(msg["k"]["v"])
        complete    =       msg["k"]["x"]
        
        self.current_price = close
        self.event_time = event_time

    def new_signal(self) -> bool:
        """Read new signal from database

        Returns:
            bool: whether new trading signal is available
        """
        
        new_signal_time = self.df_signal['creation_time'].iloc[0]
        new_signal = self.df_signal['position'].values[0]
        
        # print("Latest signal time:  {}".format(self.signal_time))
        # print("New signal time : {}".format(new_signal_time))
        
        if (new_signal_time != self.signal_time):
                    
            self.signal_time = new_signal_time
            self.signal = new_signal

            self.logger.info("New Signal Available, Signal is {}".format(self.signal))
            print("New Signal is: ", self.signal)
            return True
        
        #self.logger.info("No New Signal Available yet")
        
        return False
        
    
    def get_signal_data(self) -> None:
        """Reading new signal data from database.
        """
        
        df_signal = pd.read_parquet(self.signal_loc)
            
        df_signal = df_signal[df_signal['ticker'] == self.symbol].copy()
        
        # Get latest signal data
        self.df_signal = df_signal[df_signal['creation_time'] == df_signal['creation_time'].max()].copy()

        self.logger.info("Reading Signal data from database")
        
    def manage_position(self):
        """Manage Position and Trades, using
        the Binance ThreadedWebsocketManager callback.
        """
        
        # (1) Target/Stop Prices
        if self.position == TraderSide.BUY:
            
            target_reached = self.entry_price is not None and self.current_price >= self.target_price
            stop_reached = self.entry_price is not None and self.current_price <= self.stop_price
                        
            if (target_reached) or (stop_reached):
                
                
                if target_reached:
                    info = "Target Price has been reached Current Price: {} - Target Price: {}".format(self.current_price, self.target_price)
                elif stop_reached:
                    info = "Stop Price has been reached Current Price: {} - Stop Price: {}".format(self.current_price, self.stop_price)
                
                self.logger.info(info)
                print(info)
                         
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                
                self.position = TraderSide.NEUTRAL
                self.logger.info("GOING NEUTRAL AND STOP")
                
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                
   
        
        if self.position == TraderSide.SELL:
            
            target_reached = self.entry_price is not None and self.current_price <= self.target_price
            stop_reached = self.entry_price is not None and self.current_price >= self.stop_price
            
            if (target_reached) or (stop_reached):
                

                
                if target_reached:
                    info = "Target Price has been reached Current Price: {} - Target Price: {}".format(self.current_price, self.target_price)
                elif stop_reached:
                    info = "Stop Price has been reached Current Price: {} - Stop Price: {}".format(self.current_price, self.stop_price)
                    
                self.logger.info(info)
                print(info)
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                
                self.position = TraderSide.NEUTRAL
                self.logger.info("GOING NEUTRAL AND STOP")
                
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                
            
        
        # (2) Max Number of trades per session
        # stop trading session
        if self.trades >= self.max_trades: # stop stream after 10 trades per day
            
            # Stop Streaming
            self.twm.stop()
            
            self.logger.info("More than {} Trades executed. Stoping traiding execution.".format(self.max_trades))
            
                
            if self.position == TraderSide.BUY:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                
                
            elif self.position == TraderSide.SELL:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                
            else: 
                self.logger.info("STAYING NEUTRAL")
                print("STAYING NEUTRAL")
                
            self.position = TraderSide.NEUTRAL
            self.logger.info("GOING NEUTRAL AND STOP")    
            
        
        # (3) Max Holding Time
        elif self.trade_entry_time is not None:
            
            info = "Trading Start Time was : {} ".format(self.trade_entry_time)
            #print(info)
            #self.logger.info(info)
            
            current_trading_time = datetime.utcnow() - pd.to_datetime(self.trade_entry_time)
            max_trading_time = timedelta(minutes = self.max_trade_time)
            
            if current_trading_time >= max_trading_time:
                            
                info = "More than max time time allowed {} Minutes. Closing Trade".format(self.max_trade_time)
                self.logger.info(info)
                print(info)
                
                if self.position == TraderSide.BUY:
                    
                    order = self.client.create_order(symbol = self.symbol, 
                                                     side = "SELL", 
                                                     type = "MARKET", 
                                                     quantity = self.units)
                    
                    self.report_trade(order, "GOING NEUTRAL AND STOP")
                    
                    
                    
                elif self.position == TraderSide.SELL:
                    
                    order = self.client.create_order(symbol = self.symbol, 
                                                     side = "BUY", 
                                                     type = "MARKET", 
                                                     quantity = self.units)
                    
                    self.report_trade(order, "GOING NEUTRAL AND STOP")
                    
                else: 
                    self.logger.info("STAYING NEUTRAL")
                    print("STAYING NEUTRAL")
                    
                self.logger.info("GOING NEUTRAL AND STOP")
                self.position = TraderSide.NEUTRAL
                self.trade_entry_time = None
        
    
    def ml_signal_trader(self):
        """
        For any new ML based signal execute a new trade or keep position. 
        """
        
        # Get Latest Trading Signals
        self.get_signal_data()
        
        # Log Status every two minutes
        current_timestamp = datetime.now()
        log_info = (current_timestamp.minute % 2 == 0) \
                    and ((current_timestamp.second == 0) \
                    or (current_timestamp.second == 1))
         
        # Resync time to avoid server error           
        resync_time = (current_timestamp.minute % 20 == 0) \
                    and ( (current_timestamp.second == 0) \
                    or (current_timestamp.second == 1))
                    
        if resync_time:
            os.system('w32tm/resync')       
                    
        if log_info:
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            info = """\n{} TRADING LOG for {} 
                        | Price is : {} 
                        | Position is: {}
                        | Last Signal is : {} """.format(timestamp,
                                                         self.symbol, 
                                                         self.current_price, 
                                                         self.position, 
                                                         self.signal)
            print(info)
            self.logger.info(info)
        
        if self.new_signal():
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            
            info = "{} New Signal available".format(timestamp)
            print(info)
            self.logger.info(info)
            
            self.execute_trades()
                
                
   
    def calculate_target_stop(self) -> None:
        """Define Target and Stops trails
        https://github.com/binance/binance-spot-api-docs/blob/master/faqs/trailing-stop-faq.md
        """
        
        self.target_price = self.entry_price + self.target_usdt * self.position
        
        self.stop_price = self.entry_price - self.stop_usdt * self.position
        
        
        info = "Entry Price is {} | Target Price is {} | Stop Price is {}".format(self.entry_price, 
                                                                                  self.target_price, 
                                                                                  self.stop_price)
        self.logger.info(info)
        print(info)
        print(100 * "-" + "\n")
    
    def execute_trades(self) -> None:
        """
        Excecute Trades.
        """ 
        
        if self.signal == TraderSide.BUY: # if signal is long -> go/stay long
            
            if self.position == TraderSide.SELL:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING CLOSING SHORT TRADE") 
                print("GOING CLOSING SHORT TRADE")
                
                time.sleep(0.1)
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG") 
                print("GOING LONG")
                

            elif self.position == TraderSide.BUY:
                
                self.logger.info("STAYING LONG") 
                print("STAYING LONG")               
            
            elif self.position == TraderSide.NEUTRAL:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                                 
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG")
                print("GOING LONG")
               
            # Set position to LONG
            self.position = TraderSide.BUY  
            
            # Calculate Target and Stop            
            self.calculate_target_stop()
            
                 
        elif self.signal == TraderSide.SELL: # if signal is short -> go/stay long
            
            if self.position == TraderSide.NEUTRAL:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
            
                
            elif self.position == TraderSide.BUY:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
                time.sleep(0.1)
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT")
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
                
            
            elif self.position == TraderSide.SELL:
                
                self.logger.info("STAYING SHORT")
                print("STAYING SHORT")
                
            # Set position to SHORT  
            self.position = TraderSide.SELL
            
            # Calculate Target and Stop
            self.calculate_target_stop()

        elif self.signal == TraderSide.NEUTRAL: # if signal is neutral -> close any trade
            
            if self.position == TraderSide.BUY:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
            elif self.position == TraderSide.SELL:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
            
            # Set position to NEUTRAL 
            self.position = TraderSide.NEUTRAL
        
        
    def report_trade(self, order: dict, going :str) -> None: 
        """Report Trade to database.
        
        Args:
            order (dict): oder details.
            going (str): order type str.
        """
        # extract data from order object
        print(order)
        
        self.trade_side = order["side"]
        self.trade_orderId = order["orderId"]
        self.trade_entry_time = pd.to_datetime(order["transactTime"], unit = "ms")
        self.trade_base_unit = float(order["executedQty"])
        self.trade_qty = float(order["cummulativeQuoteQty"])
        self.entry_price = float(order["fills"][0]['price'])
        
        # calculate trading profits
        self.trades += 1
        
        if self.trade_side  == "BUY":
            
            self.trade_values.append(-self.trade_qty)
            
        elif self.trade_side  == "SELL":
            
            self.trade_values.append(self.trade_qty) 
        
        if self.trades % 2 == 0:
            
            self.trade_profit = round(np.sum(self.trade_values[-2:]), 3) 
            self.cum_profits = round(np.sum(self.trade_values), 3)
            
            # Write Trading data when trade is closed
            self.generate_trade_data()
            self.write_trade_data_parquet()
            self.write_trade_data_mongodb()
            
        else: 
            
            self.trade_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
            # if new trade has been submitted
            self.trade_id += 1
            
            # Write Trading data
            self.generate_trade_data()
            self.write_trade_data_parquet()
            self.write_trade_data_mongodb()
        
        # print trade report
        info_trading_side = "{} | {}".format(self.trade_entry_time, going)
        info_transaction_details = "{} | self.trade_base_unit = {} | self.trade_qty = $ {} | Price = $ {} ".format(self.trade_entry_time, 
                                                                                                                   self.trade_base_unit, 
                                                                                                                   self.trade_qty, 
                                                                                                                   self.entry_price)
        info_profit_details = "{} | Profit = {} | CumProfits = {} ".format(self.trade_entry_time, self.trade_profit, self.cum_profits)
        
        print(100 * "-" + "\n")
        for info in [info_trading_side, info_transaction_details, info_profit_details]:
            print(info) 
            self.logger.info(info)
        print(100 * "-" + "\n")
    

    def generate_trade_data(self) -> None:
        """Generate Trade Data
        """
        
        self.trade_data = {
            'trading_session_id' : [self.trading_session_id],
            'tradeid' : [self.trade_id] ,
            'trade_orderID' : [self.trade_orderId],
            'trade_qty' : [self.trade_qty], 
            'trade_base_unit' : [self.trade_base_unit],
            'trade_profit' : [self.trade_profit],
            'trade_entry_time': [self.trade_entry_time], 
            'trade_entry_price' : [self.entry_price],
            'cum_profits' : [self.cum_profits],
            'signal': [int(self.signal)],
            'side' :[self.trade_side]
                      }
        
        self.trade_data_mongodb = {
            'trading_session_id' : self.trading_session_id,
            'tradeid' : self.trade_id ,
            'trade_orderID' : self.trade_orderId,
            'trade_qty' : self.trade_qty, 
            'trade_base_unit' : self.trade_base_unit,
            'trade_profit' : self.trade_profit,
            'trade_entry_time': self.trade_entry_time, 
            'trade_entry_price' : self.entry_price,
            'cum_profits' : self.cum_profits,
            'signal': int(self.signal),
            'side' :self.trade_side
                      }
   
    
    def write_trade_data_parquet(self) -> None:
        """
        Write trading data to database.
        """  
        
        df_to_add = pd.DataFrame(self.trade_data)
        
        if os.path.exists(self.trading_data_path):
            
            # Read available database
            df = pd.read_parquet(self.trading_data_path)
            
            # Concat new data
            df = pd.concat([df, df_to_add], ignore_index= True)
            
            # Store         
            df.to_parquet(self.trading_data_path)
        
        else:
            
            df_to_add.to_parquet(self.trading_data_path)
            
    def write_trade_data_mongodb(self) -> None:
        """
        Write data to Mongo DB database.
        """  
           
        x = self.mongodb_collection.insert_one(self.trade_data_mongodb)
        
        self.logger.info(f"Sending data to MongoDB | Status: {x}")
        
    
    def generate_trading_data_path(self) -> None:
        """Generate trading data location path
        """
        
        os.makedirs(os.path.join(self.database_loc,  
                                self.symbol) , 
                    exist_ok=True)
        
        self.trading_data_path = os.path.join(self.database_loc, 
                                             self.symbol, 
                                             f"Trading_data_{self.trading_session_id}.parquet")
        
        info = f"Storing trading data at {self.trading_data_path}"
        self.logger.info(info)
        print(info)
        
    
    
    def generate_trading_sessions_id(self) -> None:
        """Create trading session id
        """
        
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.trading_session_id = self.symbol + "-" + now
          
    def start_trading(self) -> None:
        """Main method to execute trading for a given symbol. 
        """
        
        try:         
            
            # Generate Trading Session ID
            self.generate_trading_sessions_id()
            
            # Generate Path location to log Trading data
            self.generate_trading_data_path()
                        
            # Start Symbol price Streaming
            self.start_streaming()
            
            info = """\nTRADING LOG for {}
            | Binance Test Net 
            | Time: {} | : Price : $ {}""".format(self.symbol, 
                                                    self.event_time, 
                                                    self.current_price)
            
            # Try getting a new Trading Signal from Database
            print(info)     
            self.logger.info(info)
            
            # Look for a new signal and trade
            while True:
                
                self.manage_position()
                
                time.sleep(1)
                
                self.ml_signal_trader()
                
        except Exception as e:

                
                print(e)
                self.logger.error(e)
                substring = 'Timestamp for this request is outside of the recvWindow.'
                if substring in e.message:
                    os.system('w32tm/resync')  
        
                self.twm.stop()