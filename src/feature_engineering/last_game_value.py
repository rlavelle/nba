import pandas as pd

from src.feature_engineering.base import BaseFeature


class LastGameValueFeature(BaseFeature):
    def __init__(self,
                 source_col: str = 'points',
                 group_col: tuple[str] = ('player_id',)):
        self.source_col = source_col
        self.group_col = group_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_1g'

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(list(self.group_col))[self.source_col].shift(1)
