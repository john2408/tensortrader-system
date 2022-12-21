import pandas as pd
import numpy as np


def get_trades(data, close_column, signal_column):
    """Function to generate trade details
    """
    trades = pd.DataFrame()
    current_position = 0
    entry_time = ''

    for i in data.index:

        new_position = data.loc[i, signal_column]

        if new_position != current_position:    

            if entry_time != '':                   
                entry_price = data.loc[entry_time, close_column]
                exit_time = i
                exit_price = data.loc[exit_time, close_column]
                trade_details = pd.DataFrame([(current_position,entry_time, entry_price, exit_time,exit_price)])
                trades = trades.append(trade_details,ignore_index=True)  
                entry_time = ''            

            if new_position != 0:
                entry_time = i
            current_position = new_position


    trades.columns = ['Position','Entry Time','Entry Price','Exit Time','Exit Price']
    trades['PnL'] = (trades['Exit Price'] - trades['Entry Price']) * trades['Position']
    return trades

def get_analytics(trades):
    """Function to generate strategy analytics
    """

    analytics = pd.DataFrame(index=['Strategy'])
    # Number of long trades
    analytics['num_of_long'] = len(trades.loc[trades.Position==1])
    # Number of short trades
    analytics['num_of_short'] = len(trades.loc[trades.Position==-1])
    # Total number of trades
    analytics['total_trades'] = analytics.num_of_long + analytics.num_of_short
    
    # Gross Profit
    analytics['gross_profit'] = trades.loc[trades.PnL>0].PnL.sum()
    # Gross Loss
    analytics['gross_loss'] = trades.loc[trades.PnL<0].PnL.sum()

    # Net Profit
    analytics['net_profit'] = trades.PnL.sum()

    # Profitable trades
    analytics['winners'] = len(trades.loc[trades.PnL>0])
    # Loss-making trades
    analytics['losers'] = len(trades.loc[trades.PnL<=0])
    # Win percentage
    analytics['win_percentage'] = 100*analytics.winners/analytics.total_trades
    # Loss percentage
    analytics['loss_percentage'] = 100*analytics.losers/analytics.total_trades
    # Per trade profit/loss of winning trades
    analytics['per_trade_PnL_winners'] = trades.loc[trades.PnL>0].PnL.mean()
    # Per trade profit/loss of losing trades
    analytics['per_trade_PnL_losers'] = trades.loc[trades.PnL<=0].PnL.mean()

    return analytics.T

if __name__ == "__main__":

    get_trades(data, close_column, signal_column)
    get_analytics(trades)