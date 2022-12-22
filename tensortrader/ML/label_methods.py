import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
import os
import pandas_ta as ta

from utils.utils import *


class TechnicalSignal():
  """Calculate trading signal based on a 
  technical indicator using the pandas_ta
  libraray.
  """
  def __init__(self,  
              technical_indicator: str) -> None:
    self.technical_indicator = technical_indicator
    self.signal_strategy = []


  def run(self, 
          data: pd.DataFrame, 
          **kwargs) -> pd.DataFrame:
    
    strategy_params = {}
    for key , value in kwargs.items():
        strategy_params[key] = value

    self.define_signal_strategy(**strategy_params) 

    data = self.calculate_technical_indicator(data).copy()

    return self.calculate_signal(data).copy()


  def calculate_signal(self, df: pd.DataFrame) -> pd.DataFrame:

    if self.technical_indicator == 'awesome_os':
      # When crossing from negative to positive --> buy
      # wehn crossing from positive to negaive --> sell

      # some logic 

      return df 
    elif self.technical_indicator == 'macd_os':
      # some logic

      return df
    else:
      ValueError(f"Stragey {self.technical_indicator} is not available")

  def define_signal_strategy(self, **kwargs) -> list:

    strategy_params = {}
    for key , value in kwargs.items():
        print("key: ", key, " value", value)
        strategy_params[key] = value

    if self.technical_indicator == 'awesome_os':
      self.signal_strategy = [{'kind': 'ao', 
              'fast' : strategy_params.get('fast', 5), 
              'slow': strategy_params.get('slow', 34), 
              'offset': strategy_params.get('offset', 1)}]
    elif self.technical_indicator == 'macd_os':
      self.signal_strategy = [{'kind': 'ao', 
              'fast' : strategy_params.get('fast', 12), 
              'slow': strategy_params.get('slow', 26), 
              'signal': strategy_params.get('signal', 9), 
              'offset': strategy_params.get('offset', 1)}]
    else:
      ValueError(f"Stragey {self.technical_indicator} is not available")


  def calculate_technical_indicator(self,
                      data: pd.DataFrame, 
                      ):
    """Function to calculate technical indicators

    Args:
        data (pd.DataFrame): data containing ticker information

    Returns:
        pd.DataFrame: df containing technical indicators data
    """


    # Ref: https://github.com/twopirllc/pandas-ta/blob/main/examples/PandasTA_Strategy_Examples.ipynb

    dfs = []

    for ticker in data['Ticker'].unique():

        _df = data[data['Ticker'] == ticker].copy()

        print("Calculating Technical Indicators for ticker", ticker)

        MNQ_strategy = ta.Strategy(
            name="TA Strategy",
            description="Technical indicators Strategy",
            ta = self.signal_strategy
        )

        # Run it Technical Indicators Strategy
        _df.ta.strategy(MNQ_strategy)

        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)


class ReturnSignal():
  """Calculate a signal based on the 
  return of the Close Price of a given Ticker. 

  It calculates the lag return using the following
  formula . Given to prices P(t) and P(t+n), 
  the normalized nth return 
  r_nth(t+n) is:

  r_nth(t+n) = ((P(t+n)/P(t)) - 1)**(1/n)  - 1

  Then the return value is shifted to the 
  t-th position. Using pd.Series.shift(-n)
  For which then any ML Regression problem can 
  be trained on. 
  """

  def __init__(self, 
          return_lag: list,
          target_col_name: list,
          long_short : list,
          data : pd.DataFrame,
          timestamp_col: str,
          variable: str, 
          span_volatility: int,
          outlier_cutoff: float, 
          return_type: str) -> None:
    """Contructor. 

    Args:
        return_lags (list): list of return lags
        target_col_name (list): list of target column names
        long_short (list): list of long and short target values
        data (pd.DataFrame): input data frame of the form OHLC
        date_col (str): timestamps column
        variable (str): target variable
        span_volatility (int): Specify decay in terms of span. Defaults to 100.
        outlier_cutoff (float, optional): return outliers cutoff. Defaults to 0.01.
    """

    self.return_lag = return_lag
    self.target_col_name = target_col_name
    self.long_short = long_short
    self.data = data
    self.timestamp_col = timestamp_col
    self.variable = variable
    self.span_volatility = span_volatility
    self.outlier_cutoff = outlier_cutoff
    self.return_type = return_type


  def run(self) -> pd.DataFrame:
    """Calculate Target returns per ticker

    Returns:
        pd.DataFrame: input data frame OHLC with addtional target variable
    """

    dfs = []

    for ticker in self.data['Ticker'].unique():

        print("Calculating returns for ticker", ticker)

        _df = self.data[self.data['Ticker'] == ticker].copy()
        _df = self.calculate_returns(_df, 
                                    self.variable, 
                                    self.return_lag, 
                                    self.timestamp_col, 
                                    self.target_col_name,
                                    self.outlier_cutoff, 
                                    self.return_type)

        _df =  _df.assign(threshold = self.calculate_volatility(_df))

        _df =  _df.assign(label = self.calculate_signals(_df, self.long_short))
        
        dfs.append(_df)

    return pd.concat(dfs, ignore_index=True)

  def calculate_signals(self, df : pd.DataFrame, 
                        long_short : list) -> pd.Series:
    """Calculate trading signals
    1: long
    0: netural 
    -1: short

    Args:
        df (pd.DataFrame): _description_
        long_short (list): list of long/short strategy

    Returns:
        pd.Series: trading signals
    """

    print("Getting signals")

    long = long_short[0]
    short = long_short[1] * -1

    signals = np.where(df[self.target_col_name] > df['threshold']*long , 1 , 0)
    signals = np.where(df[self.target_col_name] < df['threshold']*short, -1 , signals)
    
    return signals
    

  def calculate_volatility(self, df):
    """Calculate volatility using
    exponentially weighted standard deviation. 

    Returns:
        pd.Series: ewm Standard deviation
    """

    print("Getting volatility")

    returns = df[self.target_col_name]

    # Estimate rolling standard deviation
    ewm_std_returns = returns.ewm(span =  self.span_volatility).std()

    return ewm_std_returns


  def calculate_returns(self,
                      data: pd.DataFrame, 
                      variable: str, 
                      return_lag: list, 
                      timestamp_col : str, 
                      target_col_name: list,
                      outlier_cutoff : float, 
                      return_type: str) -> pd.DataFrame:
    """Calculate returns base on a target variable, 
    and append ML regression target variable to the
    right order in the OHLC dataframe.

    Args:
        data (pd.DataFrame): data containing ticker information
        variable (str): target variable to calculate returns for
        return_lag (list): list of lags
        date_col (str, optional): Column holding ticker timestamps. 
        target_col_name (list): list of target column names
        outlier_cutoff (float, optional): outlier cutoff.

    Returns:
        pd.DataFrame: Dataframe containing returns features
    """
    
    if return_type == 'simple':
      returns = (data.set_index([timestamp_col])[variable]
                .sort_index() # Sort by Date
                .pct_change(return_lag) # Calculate percentage change of the respective lag value
                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                        upper=x.quantile(1-outlier_cutoff))) # Cutoff outliers
                .add(1) # add 1 to the returns
                .pow(1/return_lag) # apply n root for n = lag
                .sub(1) #substract 1
                .shift(-return_lag) # Shift Target variable to current candle --> To allow ML Regression Training
                .to_frame(target_col_name)  )          
    elif return_type =='log':
      returns = (data.set_index([timestamp_col])[variable]
                .sort_index() # Sort by Date
                .pct_change(return_lag) # Calculate percentage change of the respective lag value
                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                        upper=x.quantile(1-outlier_cutoff))) # Cutoff outliers
                .log()
                .add(1) # add 1 to the returns
                .pow(1/return_lag) # apply n root for n = lag
                .sub(1) #substract 1
                .shift(-return_lag) # Shift Target variable to current candle --> To allow ML Regression Training
                .to_frame(target_col_name)  )  



    data = data.set_index([timestamp_col]).join(returns).dropna()
    data.reset_index(inplace = True)

    return data

  def calculate_return_signal():
    pass


class TripleBarrierMethod():
  """Calculate Triple Barrier Labels based
    on profit target and stop loss strategy.
    As well as dynamic volatility as variable
    reference for selection of targeted horizontal
    barriers.

    Excepted input is pandas dataframe with 
    candle bars data of the form
    OHLC. Index are timestamps. 

    Run method generates a Dataframe containing 
    the a label for every row (candle bar). 

    1: Profitable trade
    0: No loss no gain
    -1: Stop loss

    Ref: https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e
  """


  def __init__(self, data, ticker, ptsl, delta_vertical_b, pt, 
                  delta_volatility = pd.Timedelta(hours=1), 
                  span_volatility = 100, 
                  n_jobs = 7,
                  parallel_calculation = True, 
                  max_nbytes = '1M',
                  verbose = 0):
    """Intializer method

    Args:
        data (pd.DataFrame): dataframe of the form OHLC.
        ticker (str): ticker
        ptsl (list): Profit-Stop Loss ratio [profit, stop]
        delta_vertical_b (pd.Timedelta): Trade holding time. 
        pt (int): Position Type. 1: Long. -1: Short
        delta_volatility (pd.Timedelta, optional): delta for volatility calculation. 
                                                    Defaults to pd.Timedelta(hours=1).
        span_volatility (int, optional): Specify decay in terms of span. Defaults to 100.
        n_jobs (int, optional): number of parallel jobs
        parallel_calculation (bool, optional) = True, 
        max_nbytes (str, optional): Threshold on the size of arrays passed to 
          the workers that triggers automated memory mapping in temp_folder
    """

    self._data = data
    self.ticker = ticker
    self.ptsl = ptsl
    self.delta_vertical_b = delta_vertical_b
    self.pt = pt
    self.delta_volatility = delta_volatility
    self.span_volatility = span_volatility
    self._h_barriers = None
    self.n_jobs = n_jobs
    self.parallel_calculation = parallel_calculation
    self.verbose = verbose
    self.max_nbytes = max_nbytes

  @property
  def data(self):
    return self._data

  @property
  def h_barriers(self):
    return self._h_barriers

 
  def run(self):
    """Run strategy label calculation. 

    Returns:
        pd.DataFrame: Dataframe containing the TBM label
                      for every row (candle bar). 
    """


    # Hourly Volatility
    self._data =  self._data.assign(threshold = self.calculate_volatility()).dropna()

    # Get vertical barriers timestamp
    self._data = self._data.assign(t1 = self.calculate_horizons()).dropna()

    # Get Barriers's timestamps
    events = self._data[['t1', 'threshold']].copy()

    # Add side
    events = events.assign(side=pd.Series(self.pt , events.index)) 

    # Get horizontal barriers (Target profil and stop loss)
    self._h_barriers = self.calculate_h_barriers(events)

    # Get time of barrier touches
    if self.parallel_calculation:
      touches = self.calculate_touches_parallel(events)
    else:
      touches = self.calculate_touches(events)

    # Label Barrier touches
    if self.parallel_calculation:
      touches = self.calculate_labels_parallel(touches)
    else:
      touches = self.calculate_labels(touches)

    # Add labels to data
    self._data = self._data.assign(label=touches.label.astype(int))


  def calculate_volatility(self):
      """Calculate volatility using
      exponentially weighted standard deviation. 
 
      Returns:
          pd.Series: ewm Standard deviation
      """

      print("Getting volatility")

      # 1. compute returns of the form p[t]/p[t-1] - 1
      close = self._data.Close

      # 1.1 find the timestamps of p[t-1] values
      df0 = close.index.searchsorted(close.index-self.delta_volatility)
      df0 = df0[df0>0]

      # 1.1 find the timestamps of p[t-1] values
      df0 = pd.Series(close.index[df0-1], index = close.index[close.shape[0]-df0.shape[0]:])

      # 1.3 get values by timestamps, then compute returns
      df0 = close.loc[df0.index]/close.loc[df0.values].values - 1 

      # 2. estimate rolling standard deviation
      df0 = df0.ewm(span =  self.span_volatility).std()

      return df0

  def calculate_horizons(self):
      """Get vertical barriers timestamp.

      Returns:
          pd.Series: pandas Series with vertical timestamps barriers
      """
      print("Getting horizons")

      close = self._data.Close
      t1 = close.index.searchsorted(close.index + self.delta_vertical_b)
      t1 = t1[t1 < close.shape[0]]
      t1 = close.index[t1]
      return pd.Series(t1, index=close.index[:t1.shape[0]])


  def calculate_h_barriers(self, events):
    """Get horizontal barriers

    Args:
        events: pd dataframe with columns
          t1: timestamp of the next horizon
          threshold: unit height of top and bottom barriers
          side: the side of each bet
          factors: multipliers of the threshold to set the height of 
                  top/bottom barriers

    Returns:
        pd.DataFrame: df containing the upper and lower horizontal
                      barriers
    """

    if self.ptsl[0] > 0: 
      thresh_uppr = self.ptsl[0] * events['threshold']
    else: 
      thresh_uppr = pd.Series(index=events.index) # no uppr thresh

    if self.ptsl[1] > 0: 
      thresh_lwr = -self.ptsl[1] * events['threshold']
    else: 
      thresh_lwr = pd.Series(index=events.index)  # no lwr thresh  

    return pd.DataFrame({'thresh_uppr':thresh_uppr,  \
                        'thresh_lwr':thresh_lwr},  \
                        index=events.index)

     

  def calculate_touches(self, events):
    '''
    events: pd dataframe with columns
      t1: timestamp of the next horizon
      threshold: unit height of top and bottom barriers
      side: the side of each bet
      factors: multipliers of the threshold to set the height of 
              top/bottom barriers

    Returns:
          pd.Series: time of earliest profit or stop loss 
    '''  
    print("Getting touches")

    thresh_lwr = self._h_barriers['thresh_lwr'].copy()
    thresh_uppr = self._h_barriers['thresh_uppr'].copy()

    out = events[['t1']].copy(deep=True)

    for loc, t1 in events['t1'].iteritems():

      df0 = self._data.Close[loc:t1]                              # path prices
      df0 = (df0 / self._data.Close[loc] - 1) * events.side[loc]  # path returns

      out.loc[loc, 'stop_loss'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
      out.loc[loc, 'take_profit'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit
    return out


  def calculate_touches_parallel(self, events):
    '''
    events: pd dataframe with columns
      t1: timestamp of the next horizon
      threshold: unit height of top and bottom barriers
      side: the side of each bet
      factors: multipliers of the threshold to set the height of 
              top/bottom barriers

    Returns:
          pd.Series: time of earliest profit or stop loss 
    '''  
    print("Getting touches")

    thresh_lwr = self._h_barriers['thresh_lwr'].copy()
    thresh_uppr = self._h_barriers['thresh_uppr'].copy()

    Close = self._data.Close

    def parallel_touch(Close, events, loc, t1):
      """_summary_

      Args:
          Close (pd.Series): Close prices
          events (pd.DataFrame): events data frame
          loc (datetime.datetime): candle timestamp 
          t1 (datetime.datetime): candle timestamp + horizotal barrier offset

      Returns:
          pd.DataFrame: time of earliest touch for stop and profit
      """

      out = events.loc[[loc]].filter(['t1'])
      
      df0 = Close[loc:t1]                              # path prices
      df0 = (df0 / Close[loc] - 1) * events.side[loc]  # path returns

      out['stop_loss'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
      out['take_profit'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit

      return out

    parallel_outputs = Parallel(n_jobs = self.n_jobs, max_nbytes = self.max_nbytes, verbose = self.verbose)(delayed(parallel_touch) \
                      (Close, events, loc, t1) for loc, t1 in events['t1'].iteritems())

    self.parallel_outputs = parallel_outputs

    return pd.concat(parallel_outputs)

  def calculate_labels(self, touches):
    """Assign TBM Labels

    Args:
        touches (pd.DataFrame): dataframe containing the 
            time of earliest profit or stop loss 

    Returns:
        pd.DataFrame: dataframe containing the labels
    """
    print("Getting Labels")

    out = touches.copy(deep=True)
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.iteritems():
      #print(" ", loc, " ", t)
      if pd.isnull(t):
        out.loc[loc, 'label'] = 0
      elif t == touches.loc[loc, 'stop_loss']: 
        out.loc[loc, 'label'] = -1
      else:
        out.loc[loc, 'label'] = 1
    return out

  def calculate_labels_parallel(self, touches):
    """Assign TBM Labels

    Args:
        touches (pd.DataFrame): dataframe containing the 
            time of earliest profit or stop loss 

    Returns:
        pd.DataFrame: dataframe containing the labels
    """
    print("Getting Labels")


    #out = touches.copy(deep=True)
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)

    def parellel_labels(touches, loc, t):
      """Get labels for first horizontal
      barrier touch.


      Args:
          touches (pd.DataFrame): dataframe containing the 
            time of earliest profit or stop loss 
          loc (datetime.datetime): candle bar timestamp
          t (datetime.datetime): time of touch

      Returns:
          _type_: _description_
      """

      out = touches.loc[[loc]]

      if pd.isnull(t):
        out.loc[loc, 'label'] = int(0)
      elif t == touches.loc[loc, 'stop_loss']: 
        out.loc[loc, 'label'] = int(-1)
      else:
        out.loc[loc, 'label'] = int(1)

      return out

    parallel_outputs = Parallel(n_jobs = self.n_jobs, verbose = self.verbose)(delayed(parellel_labels) \
                      (touches, loc, t) for loc, t in first_touch.iteritems())

    return pd.concat(parallel_outputs)
    
  def add_metalabel(self, y_model1):
    """Calculate Metalabels

    Args:
        y_model1 (pd.Series): Trading Strategy containing
                          1: Long trades
                          0: No trade
                          -1: Short trades

    Returns:
        np.array: metalabels:
          1: Take the trade
          0: ignore the trade
    """

    y_true = self._data['label']
    
    bin_label = np.zeros_like(y_model1)
    for i in range(y_model1.shape[0]):
        if y_model1[i] != 0 and y_model1[i]*y_true[i] > 0:
            bin_label[i] = 1  # true positive

    self._data.loc[:,'metalabel'] =  bin_label

    return self._data

  def store_data(self, output_folder_db, v_barrier_minutes):
    """Store input dataframe and labels as parquet

    Args:
        output_folder_db (str): storage folder location
        v_barrier_minutes (int): minutes for trading holding (vertical barrier)
    """

    file_name = 'Tripe_Barrier_Method_{}_ptsl_{}_vb_{}m.parquet'.format(self.ticker, '-'.join([str(x) for x in self.ptsl] ), v_barrier_minutes )
    _dir = os.path.join(output_folder_db, self.ticker)

    if not os.path.exists(_dir):
      os.mkdir(_dir)

    output_storage = os.path.join(_dir, file_name)

    print(output_storage)

    self._data.to_parquet(output_storage)


