# How to run:
# linux: export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system"
# win: ---
# cd /tensortrader/tasks/
# python price_return_calculation.py   

# Import tensortrader functions
from tensortrader.ETL.ETL_func import *
from tensortrader.ML.label_methods import *
from tensortrader.ML.models import *
from tensortrader.tasks.task_utils import create_logging
from tensortrader.constants import *

from datetime import datetime
from pathlib import Path
import yaml
import logging

def main():

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------

    # TODO: 
    # (1) label_mode: return (simple return p(t+1)/p(t)) & target_type: regression (done)
    # (2) label_mode: log_return log(p(t+1)/p(t)) & target_type: regression (to do)
    # (3) label_mode: Triple Barrier Method & target_type: classification (to do)

    # --------------------
    # Priority 1
    # --------------------
    CONF = yaml.safe_load(Path('../config/price_return.yml').read_text())
    
    # -----------------------------
    # Initial Parameters
    # -----------------------------
    n_days = CONF.get('n_days')
    years_filter = CONF.get('years_filter')
    input_folder_db = CONF.get('input_folder_db')
    label_mode = CONF.get('label_mode')
    return_type = CONF.get('return_type')
    use_resampling = CONF.get('use_resampling') 
    resampling = CONF.get('resampling')
    output_folder_db = CONF.get('output_folder_db') 
    imbalance_classes_mode = CONF.get('imbalance_classes_mode')
    time_zone = CONF.get('time_zone')
    
    # number of candles to considered for volatility
    span_volatility = CONF.get('span_volatility') 
    outlier_cutoff = CONF.get('outlier_cutoff')

    # Number of candles to hold a trade
    v_barrier_minutes = CONF.get('v_barrier_minutes')

    run_name = f"{label_mode}_{resampling}_return_{return_type}"

    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M") 

    price_return_dir = os.path.join( Path(os.getcwd()).parents[0].parents[0],
                         'logs/price_return_logs',
                         f"Price_{timestamp}_{run_name}")

    if not os.path.exists(price_return_dir):
        os.mkdir(price_return_dir)

    with open(os.path.join(price_return_dir,'config.yml'), 'w') as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Price Return data at", price_return_dir)

    LOG_FILENAME = os.path.join( price_return_dir,
                                f"{timestamp}_Price_return.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Price Return Config
    # -----------------------------
    logger.info("Reading Price Return Configuration")


    logger.info(f"""Running Price Return for Symbols: {SYMBOLS}
                    \nReading {n_days} days of historical data
                    \nFrom folder {input_folder_db}
                    \nTarget Variable is {run_name}
                    \nLabel is calculated using {label_mode}
                    """)

    # -----------------------------
    # Return Labels Parameters
    # -----------------------------
    if label_mode == 'return':
        return_lag = CONF.get('return_lag')
        long_short = CONF.get('long_short')
        timestamp_col = CONF.get('timestamp_col')
        variable = CONF.get('variable')
        target_col_name = ("{}_target_return_{}m"
                            .format(variable, 
                            int(resampling[:-3]) * return_lag))
        
        # if target_type == 'regression':
        #     target_variable = target_col_name
        # elif target_type == 'classification':
        #     target_variable = 'label'


    #-----------------------------
    # Triple Barrier Parameters
    # -----------------------------
    if label_mode == 'TBM':
        ptsl = CONF.get('ptsl')  # Profit-Stop Loss ratio
        pt = CONF.get('pt') # Position Type 1: Long, -1: Short
        delta_vertical_b = pd.Timedelta(minutes = v_barrier_minutes) 

        # Volatility Parameters
        volatility_freq = CONF.get('volatility_freq') # In minutes
        delta_volatility = pd.Timedelta(minutes = volatility_freq)
        target_variable = 'label'

        
        # For parallel labels computing
        parallel_calculation = CONF.get('parallel_calculation') 
        n_jobs = CONF.get('n_jobs') 
        max_nbytes = CONF.get('max_nbytes')

   

    # ------------------------------------------------------------------------
    # Price Return
    # ------------------------------------------------------------------------

    # -------------------------------------------------
    # 1. Data Load
    # -------------------------------------------------

    logger.info("Reading Historical data")
    Loader = DataLoader(input_folder_db = input_folder_db)

    data = Loader.load(n_days = n_days,
            symbols = SYMBOLS,
            years_filter = years_filter)

    if use_resampling:
        logger.info(f"Resampling data at {resampling}")
        data = Loader.resampling(data, resampling)

    print("\nDataset size: ")
    print(data.shape)

    # -------------------------------------------------
    # 2. Return Labels
    # -------------------------------------------------
    if label_mode == 'return':

        logger.info("Calculating Return Labels")

        R_Signals = ReturnSignal(return_lag = return_lag,
                                target_col_name = target_col_name,
                                long_short = long_short,
                                data = data,
                                timestamp_col = timestamp_col,
                                variable = variable, 
                                span_volatility= span_volatility, 
                                outlier_cutoff = outlier_cutoff, 
                                return_type = return_type)
        
        data = R_Signals.run()

        print("\n Distribution of Return Labels")
        logger.info(data.groupby(['Ticker'])['label'].value_counts())
        print(data.groupby(['Ticker'])['label'].value_counts())


    # -------------------------------------------------
    # 2. TBM Labels
    # -------------------------------------------------
    if label_mode == 'TBM':
        logger.info("Calculating Triple Barrier Methods Labels")

        dfs = []

        for ticker in SYMBOLS:
            
            print("Get TBM labels for ticker", ticker)
            
            df = data[data['Ticker'] == ticker].copy()
            df = df.set_index('timestamp').copy()

            TBM_labels = TripleBarrierMethod(df,
                                            ticker,
                                            ptsl,
                                            delta_vertical_b,
                                            pt,
                                            delta_volatility,
                                            span_volatility,
                                            n_jobs,
                                            parallel_calculation,
                                            max_nbytes )

            TBM_labels.run()
            df = TBM_labels.data.copy()
            dfs.append(df)
            del df
            
        data = pd.concat(dfs)
        del dfs

        print("\n Distribution of TBM Labels")
        print(data.groupby(['Ticker'])['label'].value_counts())
    
    logger.info(f"Price Return data generated")
    
    # Create timestamp with local timezone
    local_tz = pytz.timezone(time_zone)
    data['timestamp_local'] = data['timestamp'].apply(lambda x: utc_to_local(x, local_tz)) 

    data.to_parquet(f"{output_folder_db}/Tensor_Portfolio_{return_type}_return.parquet")
    


if __name__ == "__main__":

    #SYMBOLS = ['BTCUSDT', 'ETHUSDT']

    print("-----------------------------------------------------------")
    print("Generating Price return for symbols", SYMBOLS)
    main()
    print("\n\n")
        