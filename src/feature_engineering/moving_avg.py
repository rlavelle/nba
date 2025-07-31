import pandas as pd

from src.feature_engineering.base import BaseFeature


class SimpleMovingAvgFeature(BaseFeature):
    def __init__(self, window: int = 3, source_col: str = 'points'):
        self.window = window
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_sma_{self.window}'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id',)) -> pd.Series:
        return (df.groupby(group_col)[self.source_col]
                .rolling(self.window)
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True))


class ExponentialMovingAvgFeature(BaseFeature):
    def __init__(self, span: int = 5, source_col: str = 'points'):
        self.span = span
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_ema_{self.span}'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id',)) -> pd.Series:
        return (df.groupby(group_col)[self.source_col]
                .ewm(span=self.span)
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True))


class CumSeasonAvgFeature(BaseFeature):
    def __init__(self, source_col: str = 'points'):
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_cum_ssn_avg'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        return (
            (
                    (df.groupby(group_col)[self.source_col].cumsum()) /
                    (df.groupby(group_col)[self.source_col].cumcount() + 1)
            )
            .shift(1)
            .reset_index(level=0, drop=True)
        )
