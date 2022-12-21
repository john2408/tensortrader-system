import mlflow
import sklearn
import warnings
import os
import pandas as pd
import datetime
from functions import utils
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

def generate_testing_days(df: pd.DataFrame, 
                            n_splits: int , 
                          n_days_per_testing: int):
    """Function to generate Testing Days
    
    Args:
        n_splits (int): number of cross validations to generate
        n_days_per_testing (int): number of days per testing
    """

    if n_splits > n_days_per_testing:
        ValueError("Number of splits must greater than number of testing days")

    test_days_list = []

    unique_days = pd.to_datetime(df['Date'].unique())

    last_idx = len(unique_days) - 1

    for split in range(n_splits):

        start_idx = last_idx - n_days_per_testing - split
        end_idx = last_idx - split

        test_days =  unique_days[start_idx:end_idx]

        test_days_list.append(test_days)

    return test_days_list

def check_test_training_indeces(cv_bayesian_search, X):
    """
    Check Train-Testing Indices for a CV Object
    """

    for train_idx, test_idx in cv_bayesian_search.split(X):
        cv_key = '{}-{}:{}-{}'.format(min(train_idx), max(train_idx), min(test_idx), max(test_idx))

        print("Testing for ", cv_key)

def calculate_class_weights_from_feature(df: pd.DataFrame,
                                        Y_train: pd.DataFrame,
                                        target_variable: str, 
                                        forecast_variable: str,
                                        metric: str, 
                                        ):
    """Function to class weights from a feature column
        use the latest calculated value. 
    
    Args:
        df (pandas.Dataframe): df input data containin the features
        Y_train (pandas.DataFrame): df containing the training samples
        target_variable (str): target variable from which to get the weights
        forecast_variable (str): forecat variable for which to calculate the weights
        metric (str): column sufix to use as class weights
    """

    class_weight_column = f'{target_variable}_{metric}'

    df_class_weights = (df.groupby([target_variable])
                        .apply(lambda x: x[class_weight_column].values[-1])
                        .to_frame()
                        .reset_index()
                        .rename(columns = {0: 'class_weight'}))

    print(" calculating class weights")
    print(df_class_weights)

    df_class_weights = pd.merge(Y_train, df_class_weights, 
                            left_on = [forecast_variable] , 
                            right_on = [target_variable], 
                            how = 'left')

    classes_weights = np.array(df_class_weights['class_weight'])

    return classes_weights


class MultipleTimeSeriesCV:
    """
    Generates tuples of train_idx, test_idx pairs.
    Assumes the MultiIndex contains levels 'symbol' and 'date'.
    Purges overlapping outcomes.
    """
    
    def __init__(self,
                n_splits=3,
                train_period_length=126,
                test_period_length=21,
                lookahead=None,
                date_idx = 'date',
                shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx
    
    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days=sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead -1
            train_start_idx = train_end_idx + self.train_length + self.lookahead -1
            split_idx.append([train_start_idx, train_end_idx,
                             test_start_idx, test_end_idx])
        
        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start])
                             & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx]> days[test_start])
                            & (dates[self.date_idx] <= days[test_end])].index
            
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()
    
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits



def gridcv_xgb_model(df : pd.DataFrame,
                        target_variable : str,
                        training_date_split: datetime.datetime, 
                        param_grid: dict,
                        n_jobs : int = -2):
    """Hyperparameter Optimization for XGboost using GridSearch
    exahustive search
    
    Args:
        df (pandas.DataFrame): Input data frame containg target variable and features
        target_variable (str): Target variable to train for
        training_date_split (datetime.datetime): Split day for train and test sets 
        param_grid (dict): parameter grid
        n_jobs (int): number of cores to use
    """


    if target_variable == 'risk_type':
        X_columns = df.drop(columns = ['Date', 'Time', 'ticker','datetime', 'Time_tuple', target_variable,'entry_type','target_fh_2'] ).columns
    else:
        X_columns = df.drop(columns = ['Date', 'Time', 'ticker','datetime', 'Time_tuple', target_variable,'entry_type'] ).columns

    X_train = df[df['datetime'] < training_date_split].filter(X_columns).reset_index(drop = True)
    X_test = df[df['datetime'] >= training_date_split].filter(X_columns).reset_index(drop = True)

    Y_train = df[df['datetime'] < training_date_split].filter([target_variable]).reset_index(drop = True)
    Y_test = df[df['datetime'] >= training_date_split].filter([target_variable]).reset_index(drop = True)

    print("Target variable is: ", target_variable)
    print(Y_train[target_variable].value_counts())

    # TODO: Check for improvement in class weight functions
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y = Y_train
    )

    # TODO: Apply Grid Search on different parameeters
    xgb_model = XGBClassifier(objective="binary:logistic", 
                                booster='gbtree',
                                eval_metric='auc',
                                tree_method='hist', 
                                grow_policy='lossguide',
                                use_label_encoder=False)

    # Search Hyperparameters
    grid_search = GridSearchCV(xgb_model, param_grid= param_grid, return_train_score=True)

    # Fit model
    model = grid_search.fit(X_train, Y_train, sample_weight=classes_weights, n_jobs = n_jobs)

    # Predict training set
    Y_pred = model.predict(X_test)

    # Evaluate Predictions
    conf_matrix = confusion_matrix(Y_test, Y_pred)/len(Y_pred)
    class_accuracy = utils.cal_label_accuracy(conf_matrix)

    return model, class_accuracy, X_columns


def log_run(gridsearch: sklearn.model_selection.GridSearchCV, 
            class_accuracy: dict,
            add_params: dict,
            training_columns: list,
            model_name: str, 
            run_index: int, 
            experiment_data_folder: str, 
            conda_env: dict, 
            tags: dict,
            log_only_best: bool):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        gridsearch (sklearn.GridSearchCV): grid search object
        class_accuracy (list): class accuracy
        add_params (dict): additional parameters to log
        training_columns (list): list object containing the training columns
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        experiment_data_folder (str): folder where to store experiment data
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
        log_only_best (bool): whether logging only best result
    """
    
    cv_results = gridsearch.cv_results_

    with mlflow.start_run(run_name=str(run_index)) as run:

        timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")

        mlflow.log_param("folds", gridsearch.cv)

        if log_only_best:
            
            mlflow.log_params(gridsearch.best_params_)
        else:
            #-----------------------------------------------------------------------------
            print(" Logging parameters")
            # Get list of params used from first CV
            #params = list(cv_results['params'][0].keys())
            mlflow.log_params(cv_results['params'][run_index])

            
        #-----------------------------------------------------------------------------
        print(" Additional Parameters")
        mlflow.log_params(add_params)

        #'std_test_score'
        #-----------------------------------------------------------------------------
        print(" Logging metrics")
    
        mlflow.log_metric('mean_test_score', cv_results.get('mean_test_score')[run_index] )
        mlflow.log_metric('std_test_score', cv_results.get('std_test_score')[run_index] )
        #-----------------------------------------------------------------------------
        print(" Logging class accuracy") 
        mlflow.log_metrics(class_accuracy)  

        #-----------------------------------------------------------------------------
        print(" Logging model")        
        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=conda_env)

        #-----------------------------------------------------------------------------
        print(" Logging CV results matrix")
        if experiment_data_folder not in os.listdir():
            os.mkdir(experiment_data_folder)

        filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
        csv = os.path.join(experiment_data_folder, filename)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)
        
        mlflow.log_artifact(csv) 

        #-----------------------------------------------------------------------------
        print(" Logging Feature Importance")

        filename = "%s-%s-feature_importance.csv" % (model_name, timestamp)
        importance_path = os.path.join(experiment_data_folder, filename)

        Importance = gridsearch.best_estimator_.feature_importances_
        df_importance = pd.DataFrame({'Variable': training_columns, 'Importance':Importance})
        df_importance.sort_values(by = ['Importance'], ascending = False, inplace = True)
        df_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path) 

        #-----------------------------------------------------------------------------
        print(" Logging Features")
        filename = "%s-%s-features.txt" % (model_name, timestamp)
        features_path = os.path.join(experiment_data_folder, filename)
        features = str(training_columns)
        with open(features_path, 'w') as f:
            f.write(features)
        mlflow.log_artifact(features_path) 
       
        

        mlflow.set_tags(tags) 

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)

def log_results(gridsearch: sklearn.model_selection.GridSearchCV, 
                class_accuracy: dict,
                add_params: dict,
                training_columns: list,
                experiment_name : str, 
                experiment_data_folder: str, 
                model_name: str, 
                tags={}, 
                log_only_best=False):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        gridsearch (sklearn.model_selection.GridSearchCV): grid search object
        class_accuracy (dict): class accuracy
        add_params (dict): additional params to log
        experiment_name (str): experiment name
        training_columns (list): training columns
        experiment_data_folder (str): experiment data folder
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    conda_env = {
            'name': 'AlgoTrading',
            'channels': ['defaults'],
            'dependencies': [
                'python=3.7.7',
                'scikit-learn==0.24.2',
                {'pip': ['xgboost==1.4.2']}
            ]
        }


    best = gridsearch.best_index_

    #mlflow.set_tracking_uri("http://kubernetes.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    if(log_only_best):
        log_run(gridsearch,
                        class_accuracy, 
                        add_params,
                        training_columns,
                        model_name, 
                        best, 
                        experiment_data_folder, 
                        conda_env, 
                        tags,
                        log_only_best)
    else:
        for run_index in range(len(gridsearch.cv_results_['params'])):
            
            print("logging data for cv:", run_index)

            log_run(gridsearch,
                        class_accuracy, 
                        add_params,
                        training_columns,
                        model_name, 
                        run_index, 
                        experiment_data_folder, 
                        conda_env, 
                        tags, 
                        log_only_best)
    
    