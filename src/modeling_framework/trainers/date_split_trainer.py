from datetime import datetime

import numpy as np

from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.trainer import Trainer
import pandas as pd
from typing import Tuple, List, Any


class DateSplitTrainer(Trainer):
    def __init__(self, model: Model, metric_fn):
        super().__init__(model, metric_fn)

    def _train_and_evaluate_impl(self,
                                 df: pd.DataFrame,
                                 features: List[str],
                                 target: str,
                                 proj: bool = False,
                                 **kwargs) -> Tuple[float, List[float], List[float]]:
        """
        Standard train-test split based on date cutoff
        :param date_cutoff: Date string to split train/test data
        :param date_col: Column name containing dates
        :return: Tuple of (errors, predictions)
        """
        assert 'date_cutoff' in kwargs, 'date_cutoff not in kwargs'
        assert 'date_col' in kwargs, 'date_col not in kwargs'
        date_cutoff = kwargs['date_cutoff']
        date_col = kwargs['date_col']

        train_df, test_df = self._split_by_date(df, date_cutoff, date_col)

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)

        if proj:
            yh_train = self.model.predict(X_train)
            predictions = self.model._proj(y_train, yh_train, predictions)

        error = self.metric_fn(y_test, predictions)

        return error, predictions, y_test.to_numpy()

    def _split_by_date(self,
                       df: pd.DataFrame,
                       date_cutoff: datetime,
                       date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Helper method to split data by date"""
        train = df[df[date_col] < date_cutoff]
        test = df[df[date_col] >= date_cutoff]
        return train, test
