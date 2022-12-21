from pathlib import Path
from pyexpat import model
import yaml
import logging
from constants import *


# How to run:
# linux: export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader/tensortrader"
# win: ---
# cd /tensortrader/tasks/
# python backtesting_tasks.py   

from ETL.ETL_func import *
from ML.label_methods import *
from ML.models import *
from Features.feature_generation import FeatureEngineer
from Backtesting.backtester import Backtester

from datetime import datetime
from tasks.task_utils import create_logging


def main(symbol):

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------

    # TODO: 

    # --------------------
    # Priority 1
    # --------------------
    # Adjust all yml config files (done)
    # Create logging (done)
    # Export Charts to PDF File (done)
    # Model Storage (done)
    # Run multiple combinations of models (done) 
    # Adjust resampling for chandle stick --> 10 min, 5 min. (done)
    # Create Strategy based on Returns --> then also create labels from it (done)
    # Create XGBoost Regressor for Return Forecasting--> New Strategy (done)
    # Test models with resampling (5min, 10 min) (done)
    # Create functionality for Backtesting for (done): 
    #   (1) label_mode: return & target_type: regression
    #   (2) label_mode: return & target_type: classification
    #   (3) label_mode: TBM & target_type: classification
    # Adjust Information stored in logs (done)
    
    # Create new full test for all coins with all functionallities
        
    # Technical Strategies Pool
    # Adjust Metalables Strategies 
    # --> LSTM --> GNN = CTN + GN 
    
    
    # Adjust Multivariate Cross Validation for Oversampling Adjustment

    # --------------------
    # Priority 2
    # --------------------
    # Create Module for Feature Importance Analysis on binary classification --> https://stackoverflow.com/questions/65110798/feature-importance-in-a-binary-classification-and-extracting-shap-values-for-one
    # Add external features
    
    # --------------------
    # Priority 3
    # --------------------
    # Create Class to upload data to MONGO DB
    # Create Strategy Using Recurrent Neural Networks
    # Create Stretegy Using Convolutional Temporal Networks


    CONF = yaml.safe_load(Path('../config/backtesting.yml').read_text())
    
    # -----------------------------
    # Initial Parameters
    # -----------------------------
    n_days = CONF.get('n_days')
    years_filter = CONF.get('years_filter')
    input_folder_db = CONF.get('input_folder_db')
    label_mode = CONF.get('label_mode')
    target_type = CONF.get('target_type')
    calculate_feat_importance = CONF.get('calculate_feat_importance')


    model_type = CONF.get('model_type') 
    use_resampling = CONF.get('use_resampling') 
    resampling =  CONF.get('resampling')
    imbalance_classes_mode = CONF.get('imbalance_classes_mode')
    
    # number of candles to considered for volatility
    span_volatility = CONF.get('span_volatility') 
    outlier_cutoff = CONF.get('outlier_cutoff')

    # Number of candles to hold a trade
    v_barrier_minutes = CONF.get('v_barrier_minutes')

    run_name = f"{label_mode}_{model_type}_1min_{imbalance_classes_mode}"
    if use_resampling:
        run_name = f"{label_mode}_{model_type}_{resampling}_{imbalance_classes_mode}"

    # -----------------------------
    # Logging Config
    # -----------------------------
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M") 


    backtesting_folder = os.path.join( Path(os.getcwd()).parents[0].parents[0],
                         'backtests',
                         f"Backtest_{current_date}_{symbol}_{run_name}")

    if not os.path.exists(backtesting_folder):
        os.mkdir(backtesting_folder)

    with open(os.path.join(backtesting_folder,'config.yml'), 'w') as yaml_file:
        yaml.dump(CONF, yaml_file, default_flow_style=False)

    print("Storing Backtesting data at", backtesting_folder)

    LOG_FILENAME = os.path.join( backtesting_folder,
                                f"{current_date}_Backtester_{symbol}.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Backtesting Config
    # -----------------------------
    logger.info("Reading Backtesting configuration")

    # symbols = CONF.get('symbols') --> When reading symbols from yml
    symbols = [symbol]


    logger.info(f"""Running Backtesting for Symbols: {symbols}
                    \nReading {n_days} days of historical data
                    \nFrom folder {input_folder_db}
                    \nUsing Model {model_type}
                    \nTarget Varialbe is {run_name}
                    \nLabel is calculated using {label_mode}
                    \nThe model {model_type} will train for {target_type} variable
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
        
        if target_type == 'regression':
            target_variable = target_col_name
        elif target_type == 'classification':
            target_variable = 'label'

        if model_type == 'XGB':

            eval_metric = CONF.get('eval_metric_regress')
            objective = CONF.get('objective_regress')
            grow_policy = CONF.get('grow_policy_regress')
            booster = CONF.get('booster_regress')

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

        if model_type == 'XGB':

            eval_metric = CONF.get('eval_metric_class')
            objective = CONF.get('objective_class')
            grow_policy = CONF.get('grow_policy_class')
            booster = CONF.get('booster_class')
    
    # -----------------------------
    # Feature Engineering
    # -----------------------------
    feature_id = CONF.get('feature_id')
    conf_path = CONF.get('conf_path')

    # -----------------------------
    # Feature Selection
    # -----------------------------
    train_size = CONF.get('train_size')
    mode = CONF.get('mode')
    cols_to_drop = CONF.get('cols_to_drop')
    if label_mode == 'return':
        cols_to_drop.append(target_col_name)
    cat_columns = CONF.get('cat_columns')

    # -----------------------------
    # Model Training
    # -----------------------------
    n_splits = CONF.get('n_splits')
    n_iter = CONF.get('n_iter')
    test_period_length = CONF.get('test_period_length') # minutes   
    train_period_length = (None if 
                CONF.get('train_period_length') == "None" 
                else CONF.get('train_period_length') )                                                        
    gap = CONF.get('gap') 
    date_idx = CONF.get('date_idx') 
    n_top_features = CONF.get('n_top_features')
    max_n_estimators = CONF.get('max_n_estimators')
    model_name = CONF.get('model_name')
    backtesting_file = CONF.get('backtesting_file')

    # -----------------------------
    # Cost Analysis
    # -----------------------------
    trading_fee = CONF.get('trading_fee')


    # ------------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------------

    # -------------------------------------------------
    # 1. Data Load
    # -------------------------------------------------

    logger.info("Reading Historical data")
    Loader = DataLoader(input_folder_db = input_folder_db)

    data = Loader.load(n_days = n_days,
            symbols = symbols,
            years_filter = years_filter)

    if use_resampling:
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
                                outlier_cutoff = outlier_cutoff)
        
        data = R_Signals.run()

        # Add target Return column to be dropped before ML Training
        cols_to_drop.append(target_col_name)

        print("\n Distribution of Return Labels")
        print(data.groupby(['Ticker'])['label'].value_counts())


    # -------------------------------------------------
    # 2. TBM Labels
    # -------------------------------------------------
    if label_mode == 'TBM':
        logger.info("Calculating Triple Barrier Methods Labels")

        dfs = []

        for ticker in symbols:
            
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

    # -------------------------------------------------
    # 3. Labels Performance Analysis
    # -------------------------------------------------

    logger.info(f"Analyzing {label_mode} Labels performance")

    Signals_Backtest = Backtester(logger)

    backtesting_df_TBM = Signals_Backtest.vectorize_backtesting(ds = data.copy(),
                      modus =  'signal_validation', 
                      backtesting_folder = backtesting_folder,
                      use_adj_strategy = True,
                      v_barrier_minutes = v_barrier_minutes, 
                      trading_fee = trading_fee,
                      logger = logger)

    # -------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------
    logger.info(f"Calculating Features, using feature config number {feature_id}")
    feature = FeatureEngineer(feature_id  = feature_id, 
                            conf_path = conf_path)
    
    data = feature.calculate_features(data).copy()

    # -------------------------------------------------
    # 5. Feature Selection
    # -------------------------------------------------
    logger.info(f"Analazing most relevant features, using mode {mode}")
    data.reset_index(inplace = True)

    # Convert Categorical Columns to dummies
    data = feature.add_dummies(data, cat_columns).copy()

    # Get Columns for prediction, eliminate irrelevant columns
    predictors_list = list(set(data.columns) - set(cols_to_drop))

    # Get test/train split
    X_train, X_test, y_train, y_test = feature.train_test_split_multiple_ts(data, predictors_list, train_size, target_variable)

    # Get columns available for prediction
    predictors_list = X_train.columns
    
    if calculate_feat_importance:

        logger.info(f"Selecting top {n_top_features} features")
        feat_importance = feature.feature_selection(predictors_list = predictors_list, 
                                                X_train = X_train, 
                                                y_train = y_train, 
                                                mode = mode, 
                                                target_type = target_type)
    
        feature_selected = list(feat_importance.index[:n_top_features])
    else:
        feature_selected = list(predictors_list)

    # -------------------------------------------------
    # 6. Model Training
    # -------------------------------------------------
    
    logger.info(f"Training model using {model_type}")
    XGB_trainer = ML_trainer(train_length = train_period_length,
            test_length = test_period_length, 
            n_splits = n_splits,
            gap = gap, 
            date_idx = date_idx,
            model_type =  model_type, 
            symbols = symbols)

    print(XGB_trainer)
    logger.info(f"{XGB_trainer}")

    
    report, y_pred = XGB_trainer.fit(X_train, 
                                    X_test, 
                                    y_train, 
                                    y_test,
                                    feature_selected, 
                                    imbalance_classes_mode,
                                    target_type = target_type,         
                                    n_iter = n_iter,
                                    max_n_estimators = max_n_estimators, 
                                    eval_metric = eval_metric, 
                                    objective = objective, 
                                    grow_policy = grow_policy, 
                                    booster = booster)

    logger.info(f"Storing Model...")
    XGB_trainer.store_model(model_name = model_name, 
                            storage_folder = backtesting_folder)


    logger.info(f"Analizing model performance, {report}")

    # backtesting_df_ML = Signals_Backtest.vectorize_backtesting(ds = data.copy(),
    #                   modus =  'ML_performance', 
    #                   y_pred = y_pred,
    #                   backtesting_folder = backtesting_folder, 
    #                   use_adj_strategy  = True,
    #                   v_barrier_minutes = v_barrier_minutes, 
    #                   trading_fee = trading_fee)

    if target_type == 'regression':
    
        df_signals = (pd.DataFrame(y_pred.rename(R_Signals.target_col_name))
                .join(data.set_index('Date')
                .filter(['threshold'])))
        
        df_signals['label'] = R_Signals.calculate_signals(df_signals, long_short)
        
        backtesting_df_ML = Signals_Backtest.vectorize_backtesting(ds = data.copy(),
                            modus =  'ML_performance', 
                            y_pred = df_signals['label'],
                            backtesting_folder = backtesting_folder,
                            use_adj_strategy  = True,
                            v_barrier_minutes = v_barrier_minutes, 
                            trading_fee = trading_fee)

        
    else:    
        backtesting_df_ML = Signals_Backtest.vectorize_backtesting(ds = data.copy(),
                            modus =  'ML_performance', 
                            y_pred = y_pred,
                            backtesting_folder = backtesting_folder,
                            use_adj_strategy  = True,
                            v_barrier_minutes = v_barrier_minutes, 
                            trading_fee = trading_fee)
    
    logger.info(f"Storing backtesting results")

    Signals_Backtest.store_backtesting_results_parquet( 
                                backtesting_df =  backtesting_df_ML,
                                file_name =  backtesting_file,
                                storage_folder = backtesting_folder )
    


if __name__ == "__main__":

    SYMBOLS = [ 'BTCUSDT', 'ETHUSDT', 'LTCUSDT',  'ADAUSDT','BNBUSDT', 'BNBBTC' , 'EOSUSDT', 'ETCUSDT',
        'XMRBTC', 'TRXUSDT', 'XLMUSDT', 'IOTAUSDT',
        'MKRUSDT', 'DOGEUSDT']

    SYMBOLS = ['BTCUSDT']

    for symbol in SYMBOLS:
        print("-----------------------------------------------------------")
        print("Processing Backtesting for Symbol: ", symbol)
        main(symbol)
        print("\n\n")
        