import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import randint, uniform
import lightgbm as lgb
from tqdm.notebook import tqdm


class ColumnwiseIterativeImputer(BaseEstimator, TransformerMixin):
    """
    A custom iterative imputer that allows specifying which columns
    are used to predict the missing values for each target column.
    This version includes an initial simple imputation strategy for stability.
    """
    def __init__(self, imputation_mapping, initial_strategy='constant', fill_value=0, max_iter=10, random_state=56):
        """
        Args:
            imputation_mapping (dict): A dictionary where keys are the names of columns
                                     to impute, and values are lists of column names
                                     to use as predictors for that imputation.
            initial_strategy (str): The method for the initial simple imputation before iterating.
                                    Options: 'mean', 'median', 'most_frequent', 'constant'.
            max_iter (int): Maximum number of imputation rounds.
            random_state (int): Seed for reproducibility.
        """
        self.imputation_mapping = imputation_mapping
        self.initial_strategy = initial_strategy
        self.max_iter = max_iter
        self.fill_value = fill_value
        self.random_state = random_state
        self.imputers_ = {}

    def fit(self, X, y=None):
        """
        Fits one IterativeImputer for each entry in the mapping.
        """
        for target_col, predictor_cols in self.imputation_mapping.items():
            # Define the imputer for this specific column, passing the initial strategy.
            # This tells the imputer to start with a simple imputation before iterating.
            imputer = IterativeImputer(
                initial_strategy=self.initial_strategy,
                fill_value=self.fill_value,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            
            fit_cols = predictor_cols + [target_col]
            
            imputer.fit(X[fit_cols])
            
            self.imputers_[target_col] = imputer
        return self

    def transform(self, X):
        """
        Applies the fitted imputers to transform the data column by column.
        """

        X_transformed = X.copy()
        
        for target_col, imputer in self.imputers_.items():
            transform_cols = self.imputation_mapping[target_col] + [target_col]
            
            if not all(col in X_transformed.columns for col in transform_cols):
                missing_from_X = [col for col in transform_cols if col not in X_transformed.columns]
                print(f"Warning: Columns {missing_from_X} not found in dataframe during transform for target {target_col}. Skipping this imputation.")
                continue

            imputed_data_array = imputer.transform(X_transformed[transform_cols])
            
            imputed_block_df = pd.DataFrame(imputed_data_array, index=X_transformed.index, columns=transform_cols)
            
            X_transformed[target_col] = imputed_block_df[target_col]
            
        return X_transformed
