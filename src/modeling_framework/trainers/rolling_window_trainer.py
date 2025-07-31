from typing import List, Tuple

import numpy as np
import pandas as pd

from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.trainer import Trainer

GAME_WINDOW = 100


class RollingWindowTrainer(Trainer):
    def __init__(self, model: Model, metric_fn):
        super().__init__(model, metric_fn)

    def _train_and_evaluate_impl(self,
                                 df: pd.DataFrame,
                                 features: List[str],
                                 target: str,
                                 proj: bool = False,
                                 **kwargs) -> Tuple[float, List[float], List[float]]:
        """
        Standard rolling window split
        :param date_col: Column name containing dates
        :param window_size: number of games in training window
        :return: Tuple of (errors, predictions)
        """
        assert 'window_size' in kwargs, 'window_size not in kwargs'
        assert 'date_col' in kwargs, 'date_col not in kwargs'
        window_size = kwargs['window_size']
        date_col = kwargs['date_col']

        df = df.sort_values(date_col)
        actuals = []
        predictions = []

        for i in range(window_size, len(df) - 1):
            train = df.iloc[i-window_size:i]
            test = df.iloc[[i]]

            X_train = train[features]
            y_train = train[target]
            X_test = test[features]
            y_test = test[target]

            self.model.train(X_train, y_train)
            pred = self.model.predict(X_test)[0]

            actuals.append(y_test)
            predictions.append(pred)

        return self.metric_fn(actuals, predictions), predictions, actuals