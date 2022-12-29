import pandas as pd
import numpy as np
from binance.client import Client
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
import time
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
                target_usdt : float, 
                stop_usdt : float,
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
        self.target_usdt = target_usdt
        self.stop_usdt = stop_usdt
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        
        self.trades = 0 
        self.cum_profits = 0
        self.trade_values = []
        
        self.signal = None
        self.signal_time = None
        self.current_price = None
        self.event_time = None
        self.entry_price = None
        self.trade_time = None
        self.target_price = None
        self.stop_price = None
        
        
    def start_streaming(self) -> None:
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:

            #https://python-binance.readthedocs.io/en/latest/websockets.html
            self.twm.start_kline_socket(callback = self.manage_position,
                                        symbol = self.symbol, 
                                        interval = self.bar_length)
        
    def handle_socket_message(self, msg) -> None:
        """Handler method sample
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
        
        df_signal = pd.read_parquet(self.signal_loc)
            
        df_signal = df_signal[df_signal['ticker'] == self.symbol].copy()
        
        # Get latest signal data
        self.df_signal = df_signal[df_signal['creation_time'] == df_signal['creation_time'].max()].copy()

        self.logger.info("Reading Signal data from database")
        
    def manage_position(self, msg):
        """Manage Position and Trades, using
        the Binance ThreadedWebsocketManager callback.
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
        
        # (1) Target/Stop Prices
        if self.position == 1:
            
            target_reached = self.entry_price is not None and self.current_price >= self.target_price
            stop_reached = self.entry_price is not None and self.current_price <= self.stop_price
                        
            if (target_reached) or (stop_reached):
                
                if target_reached:
                    info = "Target Price has been reached Current Price: {} - Target Price: {}".format(self.current_price, self.target_price)
                elif stop_reached:
                    info = "Stop Price has been reached Current Price: {} - Stop Price: {}".format(self.current_price, self.stop_price)
                
                self.logger.info(info)
                print(info)
                         
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
        
        if self.position == -1:
            
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
                
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
                self.logger.info("GOING NEUTRAL AND STOP")
            
        
        # (2) Max Number of trades per session
        # stop trading session
        if self.trades >= self.max_trades: # stop stream after 10 trades per day
            
            # Stop Streaming
            self.twm.stop()
            
            self.logger.info("More than {} Trades executed. Stoping traiding execution.".format(self.max_trades))
            if self.position == 1:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            elif self.position == -1:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = -1
                
                self.logger.info("GOING NEUTRAL AND STOP")
                
            else: 
                self.logger.info("STAYING NEUTRAL")
                print("STAYING NEUTRAL")
            
            return True
        
        # (3) Max Holding Time
        if self.trade_time is not None:
            info = "Trading Start Time was : {} ".format(self.trade_time)
            #print(info)
            #self.logger.info(info)
            
            current_trading_time = datetime.utcnow() - pd.to_datetime(self.trade_time)
            max_trading_time = timedelta(minutes = self.max_trade_time)
            
            #info = "Current trading holding time {}".format(current_trading_time)
            #print(info)        
            #self.logger.info(info)
            
            if current_trading_time >= max_trading_time:
                            
                info = "More than max time time allowed {}. Closing Trade".format(self.max_trade_time)
                self.logger.info(info)
                print(info)
                
                if self.position == 1:
                    
                    order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                    self.report_trade(order, "GOING NEUTRAL AND STOP")
                    self.position = 1
                    
                    self.logger.info("GOING NEUTRAL AND STOP")
                    
                elif self.position == -1:
                    
                    order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                    self.report_trade(order, "GOING NEUTRAL AND STOP")
                    self.position = -1
                    
                    self.logger.info("GOING NEUTRAL AND STOP")
                    
                else: 
                    self.logger.info("STAYING NEUTRAL")
                    print("STAYING NEUTRAL")
                
   
    def calculate_target_stop(self) -> None:
        """Define Target and Stops trails
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
        
        
        
        if self.signal == 1: # if signal is long -> go/stay long
            
            if self.position == -1:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING CLOSING SHORT TRADE") 
                print("GOING CLOSING SHORT TRADE")
                
                time.sleep(0.1)
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG") 
                print("GOING LONG")
                
            elif self.position == 1:
                
                self.logger.info("STAYING LONG") 
                print("STAYING LONG")
                
            
            elif self.position == 0:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                                 
                self.report_trade(order, "GOING LONG")  
                
                self.logger.info("GOING LONG")
                print("GOING LONG")
            
            # Set position to LONG
            self.position = 1  
            
            # Calculate Target and Stop
            self.calculate_target_stop()
                 
        elif self.signal == -1: # if signal is short -> go/stay long
            
            if self.position == 0:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
                
            elif self.position == 1:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
                time.sleep(0.1)
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT")
                
                self.logger.info("GOING SHORT")
                print("GOING SHORT")
            
            elif self.position == -1:
                
                self.logger.info("STAYING SHORT")
                print("STAYING SHORT")
                
            # Set position to SHORT  
            self.position = -1
            
            # Calculate Target and Stop
            self.calculate_target_stop() 

        elif self.signal == 0: # if signal is neutral -> close any trade
            
            if self.position == 1:
                
                order = self.client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
                
            elif self.position == -1:
                
                order = self.client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                
                self.logger.info("GOING NEUTRAL")
                print("GOING NEUTRAL")
            
            # Set position to NEUTRAL 
            self.position = 0
        
        
    def report_trade(self, order: dict, going :str) -> None: 
        """Report Trade to database.
        
        Args:
            order (dict): oder details.
            going (str): order type str.
        """
        # extract data from order object
        side = order["side"]
        self.trade_time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        #self.entry_price = round(quote_units / base_units, 5)
        self.entry_price = float(order["fills"][0]['price'])
        
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
        info_transaction_details = "{} | Base_Units = {} | Quote_Units = $ {} | Price = $ {} ".format(self.trade_time, base_units, quote_units, self.entry_price)
        info_profit_details = "{} | Profit = {} | CumProfits = {} ".format(self.trade_time, real_profit, self.cum_profits)
        
        print(100 * "-" + "\n")
        for info in [info_trading_side, info_transaction_details, info_profit_details]:
            print(info) 
            self.logger.info(info)
        print(100 * "-" + "\n")
        
        
    def start_trading(self) -> None:
        """Main method to execute trading for a given symbol. 
        """
        # Start Symbol price Streaming
        self.start_streaming()
        
        # For for Socket to connect
        time.sleep(5)
        
        while True:
            
            info = """\nTRADING LOG for {}
                    | Binance Test Net 
                    | Time: {} | : Price : $ {}""".format(self.symbol, 
                                                          self.event_time, 
                                                          self.current_price)
            try: 
                # TRy getting a new Trading Signal from Database
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
                    
                # Look for new signal every 20 seconds
                time.sleep(20)
                
            except Exception as e:
                print(e)
                self.logger.error(e)
                break
        