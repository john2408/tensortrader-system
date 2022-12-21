import pandas as pd
import numpy as np


FEATURES_CONFIG_IDS = {1: {'ta': [{"kind": "ema", "length": 3},
                                    {"kind": "ema", "length": 15},
                                    {"kind": "ema", "length": 60},
                                    {"kind": "bbands", "length": 20},
                                    {"kind": "macd", "fast": 8, "slow": 21},
                                    {"kind":"rsi", "length":14},
                                    {"kind": "atr", "length":14},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Low",
                            'Volume_Features': False},


                        2: {'ta': [{"kind": "ema", "length": 3},
                                    {"kind": "ema", "length": 15},
                                    {"kind": "ema", "length": 60},
                                    {"kind": "bbands", "length": 20},
                                    {"kind": "macd", "fast": 8, "slow": 21},
                                    {"kind":"rsi", "length":14},
                                    {"kind": "atr", "length":5},
                                    {"kind": "atr", "length":15},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Low",
                            'Volume_Features': False},

                        3: {'ta': [{"kind": "ema", "length": 5},
                                    {"kind": "ema", "length": 30},
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "atr", "length":15},
                                    {"kind": "natr", "length":10},
                                    {"kind": "natr", "length":20},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 10,
                            'ref_variable_lags' : "entry_market", # it can be entry_type
                            'Volume_Features': True},

                        4: {'ta': [{"kind": "ema", "length": 5},
                                    {"kind": "ema", "length": 30},
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind":"rsi", "length":15},
                                    {"kind": "atr", "length":5},
                                    {"kind": "atr", "length":15},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 10,
                            # also apply low-stop diff
                            'ref_variable_lags' : "stop",
                            'Volume_Features': False},
                            

                        5: {'ta': [{"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind":"rsi", "length":15},
                                    {"kind": "atr", "length":5},
                                    {"kind": "atr", "length":15},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 10,
                            'ref_variable_lags' : "Low",
                            'Volume_Features': False},
                        
                        6: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind":"rsi", "length":15},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "atr", "length":15},
                                    {"kind": "natr", "length":10},
                                    {"kind": "natr", "length":20},
                                    {'kind':"pdist"}]
                            ,
                            'lags': None,
                            'ref_variable_lags' : "Low",
                            'Volume_Features': True},

                        7: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind":"rsi", "length":15},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":10},
                                    {"kind": "natr", "length":20},
                                    {'kind':"pdist"}]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Close",
                            'drop_lags': True,
                            'Volume_Features': True,
                            'EntryPrice_PrevClose': True,
                            'lags_diff' : [1,3,15]
                            },

                        8: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind":"rsi", "length":15},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    ]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Close",
                            'drop_lags': True,
                            'Volume_Features': True,
                            'EntryPrice_PrevClose': True,
                            'lags_diff' : [1,3,15]
                            }
                            
                            ,

                        9: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "sma", "length":5},
                                    {"kind": "sma", "length":30},
                                    ]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Close",
                            'drop_lags': True,
                            'Volume_Features': True,
                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,3,15],
                            'Return_Features': True,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': True,
                            
                            },
                    
                        10: {'ta': [
                                    {"kind": "bbands", "length": 30},
                                    {"kind": "macd", "fast": 30, "slow": 120},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":15},
                                    {"kind": "natr", "length":60},
                                    ]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Close",
                            'drop_lags': True,
                            'Volume_Features': True,
                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],
                            'Return_Features': True,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': True,
                            
                            },
                        
                        11: {'ta': [
                                    {"kind": "bbands", "length": 30},
                                    {"kind": "bbands", "length": 120},
                                    {"kind": "macd", "fast": 30, "slow": 120},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":15},
                                    {"kind": "natr", "length":60},
                                    {"kind": "natr", "length":120},
                                    {"kind": "sma", "length":60},
                                    {"kind": "sma", "length":480},
                                    ]
                            ,
                            'lags': 15,
                            'ref_variable_lags' : "Close",
                            'drop_lags': True,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],

                            'Return_Features': False,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': False,

                            'Prob_Features_Windows': (2,6), 
                            
                            },

                        11: {'ta': [
                                    {"kind": "bbands", "length": 30},
                                    {"kind": "bbands", "length": 120},
                                    {"kind": "macd", "fast": 30, "slow": 120},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":15},
                                    {"kind": "natr", "length":60},
                                    {"kind": "natr", "length":120},
                                    {"kind": "sma", "length":60},
                                    {"kind": "sma", "length":480},
                                    ]
                            ,
                            'lags': [15, 60, 120],
                            'ref_variable_lags' : ["entry_type"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],

                            'Return_Features': False,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': False,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (2,6), 
                            
                            },

                        12: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "bbands", "length": 60},
                                    {"kind": "bbands", "length": 120},
                                    {"kind": "natr", "length":20},
                                    {"kind": "natr", "length":60},
                                    {"kind": "natr", "length":120},
                                    {"kind": "sma", "length":20},
                                    {"kind": "sma", "length":240},
                                    ]
                            ,
                            'include_lags': False,
                            'lags': [15, 60, 120],
                            'ref_variable_lags' : ["entry_type"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],

                            'Return_Features': False,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': False,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (2,6),

                            },

                        13: {'ta': [
                                    {"kind": "bbands", "length": 10},
                                    {"kind": "macd", "fast": 5, "slow": 30},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":10},
                                    {"kind": "natr", "length":20},
                                    ]
                            ,
                            'include_lags': True,
                            'lags': [15,16,17,18,19,20],
                            'ref_variable_lags' : ["entry_type"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],

                            'Return_Features': True,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': True,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (4,8),

                            },

                        14: {'ta': [
                                    {"kind": "sma", "length":15},
                                    {"kind": "sma", "length":60},
                                    
                                    ]
                            ,
                            'include_lags': True,
                            'lags': [15],
                            'ref_variable_lags' : ["entry_type"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],

                            'Return_Features': True,
                            'return_lags': [1,15,30,60],
                            'Momentum_Features': False,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (2,6),

                            },

                        15: {'ta': [
                                    {"kind": "sma", "length":15},
                                    {"kind": "sma", "length":60},
                                    {"kind": "atr", "length":3},
                                    {"kind": "atr", "length":5},
                                    {"kind": "natr", "length":10},
                                    {"kind": "natr", "length":20}
                                    ]
                            ,
                            'include_lags': True,
                            'lags': [1,3,5,15,30],
                            'ref_variable_lags' : ["Close"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],
                            'binary_lags': True,

                            'Return_Features': True,
                            'return_lags': [1,5,15,30],
                            'Momentum_Features': False,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (2,6),

                            'forecast_variable':'return_5m',
                            'forecast_shift': 5

                            }, 

                         16: {'ta': [
                                       {"kind": "ema", "length": 3},
                                        {"kind": "ema", "length": 15},
                                        {"kind": "ema", "length": 60},
                                        {"kind": "bbands", "length": 20},
                                        {"kind": "macd", "fast": 8, "slow": 21},
                                        {"kind":"rsi", "length":14},
                                        {"kind": "atr", "length":14},
                                        {'kind':"pdist"}
                                    ]
                            ,
                            'include_lags': True,
                            'lags': [1,3,5,15,30],
                            'ref_variable_lags' : ["Close"], # entry_type, risk_type
                            'drop_lags': False,

                            'Volume_Features': True,
                            'Volume_Windows': (5,60),
                            'Volume_Returns': True, 
                            'Volume_Returns_lags': [5, 15],
                            'Volume_Returns_binary': True,

                            'EntryPrice_PrevClose': False,
                            'lags_diff' : [1,15,30,60],
                            'binary_lags': True,

                            'Return_Features': True,
                            'return_lags': [1, 5, 15, 60, 240],
                            'Momentum_Features': False,

                            'use_prob_features': True,
                            'probability_features' : ['entry_type', 'risk_type'],
                            'Prob_Features_Windows': (2,6),

                            'forecast_variable':None,
                            'forecast_shift': 5

                            }
                        
                        
                        }
