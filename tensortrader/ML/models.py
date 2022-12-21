from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

# Oversampling
# Ref: https://arxiv.org/pdf/1106.1813.pdf
from imblearn.over_sampling import SMOTE

#Hyperparameter Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from xgboost import XGBClassifier, XGBRegressor

import pandas as pd
import numpy as np

from os.path import join
import joblib



class ML_trainer():
    """Class to train an ML Model 
       on any data for one or multiple
       timeseries. 

       Current available models: 

       - XGB Class: Xgboost classifier for Triple Barrier Method
       - XGB Regression: Xgboost classifier for Return Forecasting
    """

    def __init__(self, 
                train_length : int,
                test_length : int, 
                n_splits : int,
                gap: int, 
                date_idx :str,
                model_type: str, 
                symbols: list) -> None:
        """
        Args:
            train_length (int): Training length
            test_length (int): Test length
            n_splits (int): number of folds
            gap (int): gap for time series ross validator
            date_idx (str): timestamp column index
            model_type (str): model type
        """
        
        self.train_length = train_length
        self.test_length = test_length
        self.n_splits = n_splits
        self.gap = gap
        self.date_idx = date_idx
        self.model_type = model_type
        self.symbols = symbols
        self.model = None

    def get_timeseries_cv(self):

        tscv = MultipleTimeSeriesCV(n_splits = self.n_splits,
                        train_period_length = self.train_length,
                        test_period_length = self.test_length,
                        gap = self.gap,
                        date_idx = self.date_idx)

        return tscv

    def __str__(self) -> str:
        return f"ML Trainer for model {self.model_type},\
                \nTest size {self.test_length}\
                \nNumber of folds {self.n_splits}\
                \nSymbols {self.symbols}"


    def fit(self, 
            X_train: pd.DataFrame, 
            X_test : pd.DataFrame, 
            y_train : np.ndarray, 
            y_test: np.ndarray,
            feature_selected: list, 
            imbalance_classes_mode: str = "class_weights",
            target_type: str = 'classification', 
            **kwargs) -> list:
        """Fit the selected ML model. 

        Args:
            X_train (pd.DataFrame): features training dataset
            X_test (pd.DataFrame): features test dataset
            y_train (np.ndarray): target variable training dataset
            y_test (np.ndarray): target variable training dataset
            feature_selected (list): selected features
            imbalance_classes_mode (str, optional): Imbalance classes method
                                "class_weights" or 
                                "oversampling". 
                                Defaults to "class_weights".

        Returns:
            list:   model: skopt.searchcv.BayesSearchCV
                    str: classification report
                    pd.Series: predicted value
        """


        print(" Dataframe training size: ", X_train.shape)
        
        if self.model_type == "XGB":

            add_params = {}
            for key , value in kwargs.items():
                print("key: ", key, " value", value)
                add_params[key] = value

            # XGboost paramters
            max_learning_rate = add_params.get('max_learning_rate', 0.05)
            max_max_depth = add_params.get('max_max_depth', 50)
            max_n_estimators = add_params.get('max_n_estimators', 500) 
            objective = add_params.get('objective', "multi:softmax") 
            booster = add_params.get('booster', "gbtree") 
            eval_metric = add_params.get('eval_metric', "auc") 
            grow_policy = add_params.get('grow_policy', "lossguide") 

            # Bayesian Cross Validation Parameters
            n_jobs = add_params.get('n_jobs', 4) # number of parallel Bayesian jobs
            n_iter = add_params.get('n_iter', 3) # iteration for Bayesian cross validation 
            random_state = add_params.get('random_state', 123) # random state bayesian cross validator

            param_grid = {
                        'learning_rate': Real(0.005, max_learning_rate, prior='log-uniform'),
                        'max_depth': Integer(3, max_max_depth, prior='log-uniform'),
                        'n_estimators': Integer(10, max_n_estimators, prior='log-uniform'),
                        }

            
            if target_type == 'classification':
                # Create classification tree model
                xgb_model = XGBClassifier(objective = objective, 
                                            booster = booster,
                                            eval_metric = eval_metric, 
                                            grow_policy = grow_policy)
                        

                # Map Target Value to allow Class weights calculation
                y_train = pd.Series(y_train).map({-1:0, 0:1, 1:2}).values
                y_test = pd.Series(y_test).map({-1:0, 0:1, 1:2}).values
                            

                if imbalance_classes_mode == "class_weights":

                    # Get timeseries cross validator
                    tscv = self.get_timeseries_cv()

                    bayes_search = BayesSearchCV(
                        xgb_model,
                        param_grid,
                        n_iter = n_iter,
                        random_state = random_state,
                        n_jobs = n_jobs,
                        cv = tscv
                    )
                    

                    # Compute class Weights
                    classes_weights = compute_class_weight(class_weight =  'balanced', classes = np.unique(y_train), y = y_train )

                    #weights = {0: classes_weights[0] , 1: classes_weights[1]}
                    weights = {0: classes_weights[0] , 1: classes_weights[1], 2: classes_weights[2]}

                    print("Training XGB Classifier Model with class weights")

                    model = bayes_search.fit(X_train.filter(feature_selected), y_train, sample_weight = pd.Series(y_train).map(weights) ) 

                    
                if imbalance_classes_mode == "oversampling":
                    # Oversampling technique
                    sm = SMOTE()

                    bayes_search = BayesSearchCV(
                        xgb_model,
                        param_grid,
                        n_iter = n_iter,
                        random_state = random_state,
                        n_jobs = n_jobs,
                        cv = self.n_splits
                    )

                    
                    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

                    print(pd.Series(y_train_res).unique())

                    print("Training XGB Classifier Model with class oversampling")

                    model = bayes_search.fit(X_train_res.filter(feature_selected), y_train_res)
                
                # Predict training set
                y_pred = model.predict(X_test.filter(feature_selected))

                # Transform Classfication Label back
                y_pred = pd.Series(y_pred, index = X_test.index).map({0:-1, 1:0, 2:1})
                y_test = pd.Series(y_test).map({0:-1, 1:0, 2:1})

                # Test accuracy
                accuracy_test = accuracy_score(y_test, y_pred)
                print("accuracy score:", round(accuracy_test, 4))

                report = classification_report(y_test, y_pred)
                print(report)
            
            elif target_type == 'regression':

                # Create classification tree model
                xgb_model = XGBRegressor(objective = objective, 
                                            booster = booster,
                                            eval_metric = eval_metric, 
                                            grow_policy = grow_policy)

                # Get timeseries cross validator
                tscv = self.get_timeseries_cv()

                # Bayesian Cross Validator Object
                bayes_search = BayesSearchCV(
                    xgb_model,
                    param_grid,
                    n_iter = n_iter,
                    random_state = random_state,
                    n_jobs = n_jobs,
                    cv = tscv
                )

                model = bayes_search.fit(X_train.filter(feature_selected), y_train ) 

                y_pred = model.predict(X_test.filter(feature_selected))

                # Transform Classfication Label back
                y_pred = pd.Series(y_pred, index = X_test.index)

                # Test accuracy
                accuracy_test = mean_squared_error(y_test, y_pred)
                print("accuracy score:", round(accuracy_test))

                report = '' # No report since it is regression

                
            self.model = model

        return report, y_pred

    def store_model(self, model_name: str, storage_folder: str ) -> None:
        """Store model as pkl data

        Args:
            model_name (str): Model Name
            storage_folder (str): Storage Location
        """
        
        model_storage_loc = join( storage_folder, model_name)
        joblib.dump(self.model, model_storage_loc)


class MultipleTimeSeriesCV:
    """
    Generates tuples of train_idx, test_idx pairs.
    Assumes the MultiIndex contains levels 'symbol' and 'date'.
    Purges overlapping outcomes.
    
    If train_period_length is not given, then 
    the trainig set will start at the first
    timestamp available.
    
    """
    
    def __init__(self,
                n_splits: int = 3,
                train_period_length : int = None,
                test_period_length : int = 10,
                gap: int = 1,
                date_idx : str = 'date',
                shuffle : bool = False):
        
        self.n_splits = n_splits
        self.gap = gap
        self.test_period_length = test_period_length
        self.train_period_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx
    
    def split(self, X, y=None, groups=None):
        """
        It generates test/trainig groups starting from 
        the last date available. 
        
        The array timestamps will be read backwards. 
        The first timestamp available has index = len(timestamps) - 1. 
        The last timestamp available has index = 0.
        
        Sample output a dataframe with only 1 time series:
        
            n_splits = 5
            test_period_length = 2    
            train_period_length = None                                                            
            gap = 5

            Output variable "split_idx":
            
            [train_start_idx, train_end_idx,
             test_start_idx, test_end_idx]
             
            --> [[3852, 6, 2, 0], [3852, 8, 4, 2], [3852, 10, 6, 4], [3852, 12, 8, 6], [3852, 14, 10, 8]]
            
            Output of the split method:
            --> 
            [   1    2    3 ... 3844 3845 3846]    [3851 3852]
            [   1    2    3 ... 3842 3843 3844]    [3849 3850]
            [   1    2    3 ... 3840 3841 3842]    [3847 3848]
            [   1    2    3 ... 3838 3839 3840]    [3845 3846]
            [   1    2    3 ... 3836 3837 3838]    [3843 3844]
        """
        
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        timestamps = sorted(unique_dates, reverse=True)
        
        if self.train_period_length is not None:
            min_train_data_available = (len(timestamps) - 1) - self.n_splits * self.train_period_length
            assert self.train_period_length >= min_train_data_available, \
                    "Train period length out of range"
        
        
        split_idx = []
        
        for i in range(self.n_splits):
            
            test_end_idx = i * self.test_period_length
            test_start_idx = test_end_idx + self.test_period_length
            train_end_idx = test_start_idx + self.gap -1
            
            if self.train_period_length is not None: 
                train_start_idx = train_end_idx + self.train_period_length + self.gap -1 
            else:
                train_start_idx = len(timestamps) - 1
                
            split_idx.append([train_start_idx, train_end_idx,
                             test_start_idx, test_end_idx])
               
        
        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > timestamps[train_start])
                             & (dates[self.date_idx] <= timestamps[train_end])].index
            test_idx = dates[(dates[self.date_idx]> timestamps[test_start])
                            & (dates[self.date_idx] <= timestamps[test_end])].index
            
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()
    
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits