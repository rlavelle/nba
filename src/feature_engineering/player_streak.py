import numpy as np
import pandas as pd

from src.feature_engineering.base import BaseFeature


class PlayerHotStreakFeature(BaseFeature):
    def __init__(self, comp_col: str, window: int = 3, source_col: str = 'points'):
        self.comp_col = comp_col
        self.window = window
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_{self.window}g_hot_streak'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        assert self.comp_col in df.columns, f'{self.comp_col} not found in provided DataFrame'

        df0 = df.copy()
        df0['avg'] = df0.groupby(group_col)[self.source_col].rolling(3).mean().shift(1).values
        df0['hot'] = np.where(df0[self.comp_col] == 0, 1, df0.avg / df0[self.comp_col])
        df0['hot'] = df0.hot.fillna(1)

        return df0.hot.reset_index(level=0, drop=True)

