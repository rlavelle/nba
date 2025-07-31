import pandas as pd

from src.feature_engineering.base import BaseFeature


class BayesPosteriorFeature(BaseFeature):
    def __init__(self, source_col: str = 'points'):
        self.source_col = source_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_bayes_post'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        df0 = df.copy()
        df0['n'] = df0.groupby([df0.player_id, df0.season])[self.source_col].cumcount() + 1

        season_stats = df0.groupby([df0.player_id, df0.season])[self.source_col].agg(
            ['mean', 'var', 'size']).reset_index()
        season_stats = season_stats.rename(columns={'mean': f'{self.source_col}_mean', 'var': f'{self.source_col}_var',
                                                    'size': f'{self.source_col}_N'})
        season_stats[f'{self.source_col}_mu0'] = season_stats.groupby('player_id')[f'{self.source_col}_mean'].shift(1)
        season_stats[f'{self.source_col}_s2bar'] = season_stats.groupby('player_id')[f'{self.source_col}_var'].shift(1)
        season_stats[f'{self.source_col}_tau20'] = season_stats[f'{self.source_col}_s2bar'] / season_stats[
            f'{self.source_col}_N']

        df0 = df0.merge(season_stats[
                            list(group_col) +
                            [f'{self.source_col}_mu0',
                             f'{self.source_col}_s2bar',
                             f'{self.source_col}_tau20']
                            ],
                        on=['player_id', 'season'], how='left')

        df0[f'{self.source_col}_mu_n'] = (
                (
                        (df0.n * df0[self.source_col]).shift(1) / df0[f'{self.source_col}_s2bar'] +
                        df0[f'{self.source_col}_mu0'] / df0[f'{self.source_col}_tau20']
                ) /
                (
                        df0.n.shift(1) / df0[f'{self.source_col}_s2bar'] + 1 / df0[f'{self.source_col}_tau20']
                )
        )

        # todo: figure out how to pass this back through incase we want the predictive distribution
        df0[f'{self.source_col}_var_n'] = 1 / (
                df0.n.shift(1) / df0[f'{self.source_col}_s2bar'] + 1 / df0[f'{self.source_col}_tau20'])

        return df0[f'{self.source_col}_mu_n'].reset_index(level=0, drop=True)
