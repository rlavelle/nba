from datetime import datetime

from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.trainer import Trainer
import pandas as pd
from typing import Tuple, List


class DateSplitTrainer(Trainer):

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

        if 'std' in kwargs:
            standardizer = kwargs['std'](train_df, features+[target])
            train_df = standardizer.transform(train_df, handle_unseen='global')
            test_df = standardizer.transform(test_df, handle_unseen='global')

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        assert X_train.shape[0] > 0, 'No training points'
        assert X_test.shape[0] > 0, 'No testing points'

        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)

        if proj:
            yh_train = self.model.predict(X_train)
            predictions = Model._proj(y_train, yh_train, predictions)

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
