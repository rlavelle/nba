from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Callable

import numpy as np
import pandas as pd

from src.modeling_framework.framework.model import Model


class Trainer(ABC):
    def __init__(self,
                 model: Model,
                 metric_fn: Callable[[np.array, np.array], float]):
        self.model = model
        self.metric_fn = metric_fn

    def train_and_evaluate(self,
                           df: pd.DataFrame,
                           features: List[str],
                           target: str,
                           **kwargs) -> Tuple[float, List[float], List[float]]:
        """
        Trains and evaluates a model, returns errors and predictions
        :param df: data which contains feature cols
        :param features: col names to be included as features
        :param target: Y variable to predict
        :param kwargs: extr arguments (like date, window size etc)
        :return: returns errors and predictions
        """
        # Common assertions/validations for all child classes
        assert len(features) > 0, "At least one feature must be provided"
        assert len(df) > 0, "DataFrame cannot be empty"
        assert len(set(features) - set(df.columns)) == 0, \
            f"not all features contained in data: {set(features) - set(df.columns)}"

        dff = df.dropna(subset=features+[target])
        return self._train_and_evaluate_impl(dff, features, target, **kwargs)

    @abstractmethod
    def _train_and_evaluate_impl(self,
                                 df: pd.DataFrame,
                                 features: List[str],
                                 target: str,
                                 proj: bool,
                                 **kwargs) -> Tuple[float, List[float], List[float]]:
        """
        Child classes must implement this method
        """
        pass
