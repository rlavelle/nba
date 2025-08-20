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
        nan_filler = df.groupby(list(group_col))[self.source_col].expanding().mean().shift(1)
        return (df.groupby(list(group_col))[self.source_col]
                .rolling(self.window)
                .mean()
                .shift(1)
                .fillna(nan_filler)
                .reset_index(level=0, drop=True))


class ExponentialMovingAvgFeature(BaseFeature):
    def __init__(self, span: int = 5, source_col: str = 'points'):
        self.span = span
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_ema_{self.span}'

    def _f(self, x):
        return 2 / (x + 1)

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id',)) -> pd.Series:
        nan_filler = (
            df.groupby(list(group_col))[self.source_col]
            .expanding()
            .apply(lambda x: x.ewm(alpha=self._f(len(x)), adjust=True)
                   .mean().iloc[-1])
        )
        return (df.groupby(list(group_col))[self.source_col]
                .ewm(span=self.span)
                .mean()
                .shift(1)
                .fillna(nan_filler)
                .reset_index(level=0, drop=True))


class CumSeasonAvgFeature(BaseFeature):
    def __init__(self, source_col: str = 'points'):
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_cum_ssn_avg'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        """
        cumulative season average
        :return: cum season avg
        """
        return (df.groupby(list(group_col))[self.source_col]
                .expanding()
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True))


class CumSeasonEMAFeature(BaseFeature):
    def __init__(self, span: int = 5, source_col: str = 'points'):
        self.span = span
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_cum_ssn_ema'

    def _f(self, x):
        return 2 / (x + 1)

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        """
        uses a decay factor of 2 / (n+1)
        :return: exponential weighted average across current observed data per season
        """
        return (df.groupby(list(group_col))[self.source_col]
                .expanding()
                .apply(lambda x: x.ewm(alpha=self._f(len(x)), adjust=False).mean().iloc[-1])
                .shift(1)
                .reset_index(level=0, drop=True))
