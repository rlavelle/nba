import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.feature_engineering.base import BaseFeature


class ARIMAMinutesFeature(BaseFeature):
    """ARIMA forecast for minutes"""

    def __init__(self, order: tuple = (5, 1, 3), min_history: int = 5):
        self.order = order
        self.min_history = min_history

    @property
    def feature_name(self) -> str:
        return 'arima_minutes'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id',)) -> pd.Series:
        result = pd.Series(np.nan, index=df.index)

        for _, group in df.groupby(group_col):
            minutes_series = group['minutes'].astype(float).tolist()
            arima_preds = [np.nan] * len(minutes_series)

            for i in range(self.min_history, len(minutes_series)):
                try:
                    model = ARIMA(minutes_series[:i], order=self.order)
                    model_fit = model.fit()
                    arima_preds[i] = model_fit.forecast()[0]
                except:
                    arima_preds[i] = np.nan

            result.loc[group.index] = arima_preds

        return result
