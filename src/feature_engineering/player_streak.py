import numpy as np
import pandas as pd

from src.feature_engineering.base import BaseFeature

# TODO: theres some strange bug if numpy is > 2.4.0 and pandas is > 2.3.3
#       where you end up with a phantom div by 0 bug that IS NOT SOLVED
#       by using a save div0 function like thats in the class
#       *** Total lost hours of my life: 3 ***
class PlayerHotStreakFeature(BaseFeature):
    def __init__(self,
                 comp_col: str,
                 window: int = 3,
                 source_col: str = 'points',
                 group_col: tuple[str] = ('player_id', 'season')):
        self.comp_col = comp_col
        self.window = window
        self.source_col = source_col
        self.group_col = group_col

        super().__init__(self.group_col)

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_{self.window}g_v_{self.comp_col}_hot_streak'

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        assert self.comp_col in df.columns, f'{self.comp_col} not found in provided DataFrame'

        df0 = df.copy()
        df0['avg'] = df0.groupby(list(self.group_col))[self.source_col].rolling(self.window).mean().shift(1).values
        df0['hot'] = np.where(df0[self.comp_col] == 0, 1, df0.avg / df0[self.comp_col])
        df0['hot'] = df0.hot.fillna(1)

        return df0.hot.reset_index(level=0, drop=True)
