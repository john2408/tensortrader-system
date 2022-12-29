"""Main module."""
import pandas as pd
import numpy as np
import os
from utils.utils import *
from ML.label_methods import *
from ETL.ETL_func import *
from datetime import datetime
import os
import logging
import dateutil


SYMBOLS = SYMBOLS[1:]

if __name__ == "__main__":

    #path_prefix = '/media/john' # "/mnt/"
    path_prefix = "/mnt"

    #--------------------------------------------------------------------
    # Data Load
    #--------------------------------------------------------------------


    input_folder_db = os.path.join(path_prefix, 'Data/Tensor_Invest_Fund/data/Cryptos/')
    initial_date_load = datetime.today() - dateutil.relativedelta.relativedelta(days = 60)
    years_filter = [2022]

    dfs = []

    for symbol in SYMBOLS:
        for year in years_filter:

            file_name = os.path.join(input_folder_db, f'{symbol}', f'{year}_{symbol}.parquet'  )
            print("Reading file", file_name)
            
            df = pd.read_parquet(file_name)
            df = df[df['Date'] >= initial_date_load].copy()
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


    if True:    

        #--------------------------------------------------------------------
        # TBM Calculation
        #--------------------------------------------------------------------

        PTSL = [ [1,1], [2,2], [2,1] ] 

        # https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model
        # Set tmp folder for joblib otherwise parallel label calculation runs into erro 
        # OSError: [Errno 28] No space left on device
        os.environ['JOBLIB_TEMP_FOLDER'] = os.path.join(input_folder_db, 'tmp')

        # Profit-Stop Loss ratio
        #ptsl = [1,1]
        v_barrier_minutes = 15
        delta_vertical_b = pd.Timedelta(minutes = v_barrier_minutes) # Vertical barrier length 
        output_folder_db = os.path.join(path_prefix, 'Data/Tensor_Invest_Fund/data/Cryptos/TBM')

        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        LOG_FILENAME = os.path.join( os.getcwd(), 'logs',  f"{current_date}_TBM_labels.log")
        logging.basicConfig(filename = LOG_FILENAME, level = logging.DEBUG, format= '%(asctime)s %(message)s', datefmt= '%m/%d/%Y %I:%M:%S %p')
        
        # Volatility Parameters
        delta_volatility = pd.Timedelta(hours=1)
        span_volatility = 100

        # Position Type 
        # 1: Long
        # -1: Short
        pt = 1
        n_jobs = -2
        max_nbytes = '0.8M'
        parallel_calculation = False

        for ptsl in PTSL:
            
            logging.info(f"Calculating labels for profit-stop ratio {ptsl}")
            

            if parallel_calculation:
                logging.info(f"using Parallel Computing {ptsl}")
            else:
                logging.info(f"using Linear Computing {ptsl}")

            print("Calculating labels for profit-stop ratio ", ptsl)

            for ticker in SYMBOLS:

                runtime_start = time.time()

                logging.info(f"Calculating labels for ticker {ticker}")

                #df_sub = data[(data['Date'] >'2022-03-25') & (data['Ticker'] == ticker)].copy()

                df_sub = data[(data['Ticker'] == ticker)].copy()

                print("Generating TBM Labels for ticker", ticker)
                print(" in the timeframe ", df_sub['Date'].min(), " - ", df_sub['Date'].max())
                print(df_sub.shape)

                df_sub = df_sub.set_index('timestamp').copy()

                try: 
                    TBM_parallel = TripleBarrierMethod(df_sub, 
                                        ticker,
                                        ptsl, 
                                        delta_vertical_b, 
                                        pt, 
                                        delta_volatility, 
                                        span_volatility, 
                                        n_jobs,
                                        parallel_calculation, 
                                        max_nbytes)

                    TBM_parallel.run()

                    print("\n")
                    print(TBM_parallel.data['label'].value_counts())
                    print("\n")

                    TBM_parallel.store_data(output_folder_db, v_barrier_minutes)
                except Exception as e:
                    logging.info(e)


                runtime = round((time.time() - runtime_start)/60, 2)
                logging.info(f" Runtime was {runtime} minutes")
