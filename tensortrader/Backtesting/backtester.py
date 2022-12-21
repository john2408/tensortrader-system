import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join
from matplotlib.figure import Figure
import fastparquet
import logging


# Tensortrader functions
from Backtesting.bt_helpers import *

import warnings
warnings.filterwarnings('ignore')


class Backtester():

    def __init__(self, logger: logging.Logger) -> None:
        """
        Args:
            logger (logging.Logger): logger object
        """
        self.logger = logger


    def trading_cost(self, 
                    Close: pd.Series, 
                    trading_fee: float ) -> float:

        # Cost Analysis
        dec_points = len(str(Close.values[0]).split(".")[1])
        spread = 2 * 1 / (10 ** dec_points) # pips == fourth price decimal
        half_spread = spread / 2 # absolute tc per trade (position change +-1)
        ptc = half_spread / Close.mean() # proportional tc per trade (position change +-1)

        # add trading fee to position trading cost
        ptc += trading_fee
        
        return ptc

    def vectorize_backtesting(self, 
                          ds: pd.DataFrame,
                          backtesting_folder :str,
                          y_pred: np.array = None,
                          test_size_performance: int = 0.3,
                          modus: str = 'signal_validation',
                          **kwargs) -> pd.DataFrame:
        """Calculate vectorized backtesting for a given trading signal 
        or the output of a trained ML model on a signal. 

        Select modus: 
        'signal_validation': Evaluates the performance of a trading signal
                                                strategy in a selected time frame. 
        'ML_performance': Evaluates the y_pred of a ML strategy.

        Args:
            ds (pd.DataFrame): Dataset containing historical data for the selected
                                tickers
            backtesting_folder: (str): Folder to store PDF Plots
            y_pred (np.array, optional): When analysing an ML Strategy, the resulting
                                        label prediction from model testing. 
                                        Defaults to None.
            test_size_performance (int, optional): Test size performance when evaluation a 
                                                    white box strategy like Triple Barrier 
                                                    Method. Defaults to 0.3.
            
            modus (str, optional): Either 'signal_validation' or 'ML_performance'. 
                                'signal_validation': Evaluates the performance of a trading signal
                                                strategy in a selected time frame. 
                                'ML_performance': Evaluates the y_pred of a ML strategy.
                                Defaults to 'signal_validation'.

        Return: 
            pd.DataFrame: Backtesting dataframe

        """
    
        add_params = {}
        for key , value in kwargs.items():
            print("key: ", key, " value", value)
            add_params[key] = value

        use_metalabels = add_params.get('use_metalabels', False)
        use_adj_strategy = add_params.get('use_adj_strategy', False)
        v_barrier_minutes = add_params.get('v_barrier_minutes', None)
        trading_fee = add_params.get('trading_fee', 0.001)
        
        
        # Get trading cost
        ptc = self.trading_cost(ds['Close'], trading_fee)
        
        # Add Sell/Buy/Neutral Decision depending on analysis type
        if modus == 'signal_validation':
            pass
            #df.loc[:,"label"] = df["label"] # Just to make explit that TBM Decisions stay the same
        if modus == 'ML_performance':
            
            df_ref = pd.DataFrame(index = y_pred.index)
            df_ref["label"] = y_pred
            
            print(ds.shape)
            ds = (ds.drop(columns = 'label')
                .set_index(['Ticker','Date']))
            
            ds = df_ref.join(ds).reset_index()
            
            print(ds.shape)
            
            
        # Columns to keep
        cols = ['Date', 'Close','Volume','label', 'Ticker']
        if use_metalabels:
            cols.append('metalabel')
        
        # Create PDF to store the plots
        pp = PdfPages(join(backtesting_folder, f"{modus}.pdf"))

        backtesting = []

        for ticker in ds['Ticker'].unique():
            
            df = ds[ds['Ticker'] == ticker].copy() 
            
            # Analyse TBM Performance in Test Set
            # if not given usingdata set split percentage of 70%
            if modus == 'signal_validation': 
                
                split_percentage = add_params.get('split_percentage',  1 - test_size_performance)
                split = int(split_percentage*len(df))
                df = df[split:].filter(cols).copy()
                df.reset_index(inplace = True)
                
            if modus == 'ML_performance':
                
                df = df.filter(cols).copy()
                df.reset_index(inplace = True)

            print("\nAnalyzing performance for ", ticker)

            # Calculate Close return
            # and 
            df.loc[:,'Return'] = df['Close'].pct_change(1)
            df.loc[:,'buy_hold'] = df["Return"].cumsum().apply(np.exp)
            
            # Calculate Cumulative Return for buy_hold and ML Strategy
            df.loc[:,'ml_return'] = df['label'].shift(1)* df['Return']
            df.loc[:,'ml_performance'] = df["ml_return"].cumsum().apply(np.exp)

            if use_adj_strategy:
                
                col_labels = 'label'
                col_name_strategy = 'label_adj'
                col_name_strategy_return = 'ml_adj_return'
                col_name_strategy_cum_return = 'ml_adj_performance'
                
                df = self.calculate_labels_return(df, 
                                col_labels,
                                col_name_strategy,
                                col_name_strategy_return, 
                                col_name_strategy_cum_return,
                                v_barrier_minutes,
                                )

            if use_metalabels and not (modus == 'ML_performance'):
                
                col_labels = 'metalabel'
                col_name_strategy = 'metalabel_adj'
                col_name_strategy_return = 'meta_return'
                col_name_strategy_cum_return = 'meta_performance'

                df = self.calculate_labels_return(df, 
                                col_labels,
                                col_name_strategy,
                                col_name_strategy_return, 
                                col_name_strategy_cum_return,
                                v_barrier_minutes,
                                )
                
            # -------------------------------------------
            # Cost Calculation
            # -------------------------------------------
            # Cost Calculation ML Strategy
            df = self.cost_analysis(df, 
                        col_name_strategy = 'label' , 
                        col_name_strategy_return = 'ml_return',
                        col_name_strategy_cum_return  = 'ml_performance', 
                        ptc = ptc)
            
            # Cost Calculation Adj Strategy or Metalabels
            if use_adj_strategy or use_metalabels:
                df = self.cost_analysis(df, 
                        col_name_strategy, 
                        col_name_strategy_return,
                        col_name_strategy_cum_return, 
                        ptc = ptc)
                
                

            # -------------------------------------------
            # Plot Backtesting Performance
            # -------------------------------------------
            
            backtest_plot = self.plot_strategy_performance(df, 
                                    use_adj_strategy, 
                                    use_metalabels, 
                                    modus, 
                                    ticker, 
                                    col_name_strategy_cum_return)
            # Save plot to PDF
            pp.savefig(backtest_plot)
            
            backtesting.append(df)

        pp.close()

        return pd.concat(backtesting)

    def store_backtesting_results_parquet(self, 
                                backtesting_df: pd.DataFrame,
                                file_name: str, 
                                storage_folder: str ) -> None:
        """Store backtesting results as parquet file

        Args:
            model_name (str): Model Name
            storage_folder (str): Storage Location
        """
        
        data_storage_loc = join( storage_folder, file_name)
        backtesting_df.to_parquet(data_storage_loc)

    def calculate_labels_return(self, 
                            df: pd.DataFrame, 
                            col_labels: str,
                            col_name_strategy: str,
                            col_name_strategy_return: str, 
                            col_name_strategy_cum_return: str,
                            v_barrier_minutes: int
                             ) -> pd.DataFrame:
        """Calculate the return and cumulative 
        return a given strategy.

        Args:
            df (pd.DataFrame): input data
            col_labels (str): column labels name
            col_name_strategy (str): Column Strategy Name
            col_name_strategy_return (str): Column Strategy Return Name
            col_name_strategy_cum_return (str): Column Strategy Cumulative Return Name
            v_barrier_minutes (int): lenght of vertical barrier (in minutes)

        Returns:
            pd.DataFrame: Input dataframe with additional 3 columns for the strategy return
        """
    
        df.loc[:,col_name_strategy]  = self.adj_ml_strategy(df[col_labels], v_barrier_minutes )
        df.loc[:,col_name_strategy_return] = df.loc[:,col_name_strategy].shift(1)* df['Return']
        df.loc[:,col_name_strategy_cum_return] = df[col_name_strategy_return].cumsum().apply(np.exp)
        
        return df


    def cost_analysis(self, 
                  df : pd.DataFrame, 
                  col_name_strategy: str, 
                  col_name_strategy_return: str,
                  col_name_strategy_cum_return: str, 
                  ptc : float) -> pd.DataFrame:
        """Cost analysis.

        Args:
            df (pd.DataFrame): backtesting dataframe
            col_name_strategy (str): Column Strategy Name
            col_name_strategy_return (str): Column Strategy Return Name
            col_name_strategy_cum_return (str): Column Strategy Cumulative Return Name
            ptc (float): Trading cost per trade

        Returns:
            pd.DataFrame: Input dataframe with net return analysis
        """
        
        prefices = {'label': '', 
                    'metalabel_adj' : '_meta', 
                    'label_adj' : '_adj'}
        
        
        col_prefix = prefices.get(col_name_strategy)
        
        # Number of trades ml strategy adjusted
        # Changing from 1 position to another one counts as two  trades
        df[f"n_trader_ml{col_prefix}"] = df[col_name_strategy].diff().fillna(0).abs() 
        
        # Spread and trading Cost
        df[f'cost_ml{col_prefix}'] = df[f'n_trader_ml{col_prefix}'] * ptc
        
        # Total cost in USDT units
        df[f'cost_ml{col_prefix}_USDT'] = df[f'cost_ml{col_prefix}'] * df['Close']
        
        # Strategy Return after cost        
        df[f'ml_strategy{col_prefix}_c'] = df[col_name_strategy_return] - df[f'cost_ml{col_prefix}']
        
        # Calculate Net Profit
        df[f'ml_strategy{col_prefix}_net'] = df[f'ml_strategy{col_prefix}_c'].cumsum().apply(np.exp)

        initial_investment = df['Close'].values[0]
        final_return = df[f'ml_strategy{col_prefix}_net'].values[-1]
        final_investment = initial_investment*(1+final_return)
        total_trading_cost = df[f'cost_ml{col_prefix}_USDT'].sum()
        performance_ml = final_investment 

        summary_str = ("""\nML Strategy: Saldo after Backtesting for {}
                size 1 unit of the Coin $USD""".format(col_name_strategy_cum_return) + 
        "\n Initial Investment: {}".format(np.round(initial_investment, 2) ) +
        "\n Final Performance: {}".format( np.round(performance_ml, 2)) + 
        "\n Performance Return: {}".format(np.round(final_return, 2)) +
        "\nNumber of Trades: {}".format( df.n_trader_ml.sum()/2) +
        "\nTotal Cost USDT: {}".format( np.round( total_trading_cost, 2)))

        self.logger.info(summary_str)

        print(summary_str)
        
        return df

    def plot_strategy_performance(self, 
                              df: pd.DataFrame, 
                              use_adj_strategy: bool, 
                              use_metalabels: bool, 
                              modus: str, 
                              ticker: str,
                              col_name_strategy_cum_return: str = None, 
                              ) -> Figure:
        """Plot Strategy Return Performance. 

        Args:
            df (pd.DataFrame): backtesting dataframe
            use_adj_strategy (bool): whether the adjusted strategy was used
            use_metalabels (bool): whether metalabel approach was used
            modus (str): backtesting modus
            ticker (str): Ticker
            col_name_strategy_cum_return (str): Adjusted strategy column name

        """

        # Reset Timestamp index
        df.reset_index(inplace = True)
        timecolumn = 'Date'

        plot = plt.figure(figsize=(12, 6))
        print(type(plot))
        plt.plot(df[timecolumn], df["buy_hold"])
        plt.plot(df[timecolumn], df["ml_strategy_net"])
        
        if use_adj_strategy or use_metalabels:
            
            plt.plot(df[timecolumn], df["ml_strategy_adj_net"])

            if modus == 'signal_validation':
                plt.legend(["buy_hold", "Signal_performance", f"Signal_{col_name_strategy_cum_return}"])
            else:
                plt.legend(["buy_hold", "ml_performance", col_name_strategy_cum_return])
        else:
            plt.legend(["buy_hold", "ml_performance"])

        plt.title(f"""Cumulative return of buy-and-hold 
                vs machine learning 
                from {df[timecolumn].min()} - {df[timecolumn].max()}, ticker {ticker}""", fontsize=11)
        plt.show()

        return plot

    def adj_ml_strategy(self, 
                    _input : np.array, 
                    v_barrier_minutes : int, 
                    verbose :int = 0) -> np.array:
        """
        Adjust trading signals based on different
        rules. 
        
        Args: 
            signals (np.array) : array of trading signals
            v_barrier_minutes (int): vertical barrier holding
                                        period
        
        Returns:
            Adjusted array of trading signals
            
        ## Unit Test
        v_barrier_minutes = 5
        _input = [-1, 1, 0, 1, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, 1, 0 , 1]
        _output = [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0 , 1]


        assert all(np.array(adj_pred) == np.array(_output)), "Error"
        """

        n_signals = len(_input)

        index = 0
        adj_pred = []

        while(index < n_signals):

            if _input[index] == 1:


                index += 1
                adj_pred.append(1)
                if index > n_signals -1:
                    break
                if verbose > 1:
                    print(index, ": old value ", _input[index], "new value", 1)

                for i in range(1, v_barrier_minutes):

                    adj_pred.append(0)
                    index += 1

                    if index > n_signals -1:
                        break
                        
                    if verbose > 1:
                        print(index, ": old value for", _input[index], "new value:", 0)


                if index < n_signals:
                    adj_pred.append(-1)
                    index += 1
                    
                    if verbose > 1:
                        print(index, ": old value for", _input[index], "new value:", -1)
                else:
                    break

            # Otherwise add 0
            if index >= n_signals:
                break

            adj_pred.append(0)        
            index +=1
            
            if verbose > 1:
                print(index, ": old value", _input[index], "new value", 0)
            
        assert len(adj_pred) == len(_input), " Different input/ouptput lengths"
            
        return adj_pred