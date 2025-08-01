import pandas as pd

from src.feature_engineering.base import BaseFeature


class BayesPosteriorFeature(BaseFeature):
    def __init__(self, ybar_col: str, source_col: str = 'points'):
        self.prior_uncertainty = 1.5
        self.source_col = source_col
        self.ybar_col = ybar_col

    @property
    def feature_name(self) -> str:
        return f'{self.source_col}_{self.ybar_col}_ bayes_post'

    def calculate(self, df: pd.DataFrame, group_col: tuple[str] = ('player_id', 'season')) -> pd.Series:
        """
        Bayesian updating for \theta (mean of source col {f}), using an assumed variance (\sigma^2) on {f}
        where y denotes the current seasons observed data. using an assumed variance means we do not need to use
        a sampler to fully estimate the posterior. Normal Normal model for \theta is selected based on the observed data

        p(\theta|\sigma^2,y) \propto N(\theta|\mu_0,\tau_0^2) * likelihood(y|\theta,\sigma^2)

        where

        \mu_0 ~ prev seasons mean {f}
        \tau_0^2 ~ var of all previously seen seasons
        \sigma^2 ~ prev season var {f}

        gives \theta ~ N(\mu_n, \k_n)

        where

        \mu_n = [(\mu_0 / \tau_0^2) + n*(ybar / \sigma^2)] / [(1 / \tau_0^2) + (n / \sigma^2)]
        \k_n = [(1 / \tau_0^2) + (n / \sigma^2)]^-1

        NOTE: when passing ybar col in, it only really makes sense to have "n" weight if its using
              all season data... i.e., sma w/ window=3 | last game metric

        :return: bayesian posterior estimations of mean {f} given current seasons observed data
        """
        assert self.ybar_col in df.columns, f'{self.ybar_col} not found in provided DataFrame'

        df0 = df.copy()
        df0['n'] = df0.groupby(list(group_col))[self.source_col].cumcount() + 1

        season_stats = df0.groupby(list(group_col))[self.source_col].agg(
            ['mean', 'var', 'size']).reset_index()
        season_stats = season_stats.rename(columns={'mean': f'{self.source_col}_mean', 'var': f'{self.source_col}_var',
                                                    'size': f'{self.source_col}_N'})

        season_stats[f'{self.source_col}_mu0'] = season_stats.groupby('player_id')[f'{self.source_col}_mean'].shift(1)
        season_stats[f'{self.source_col}_s2bar'] = season_stats.groupby('player_id')[f'{self.source_col}_var'].shift(1)

        season_stats[f'{self.source_col}_tau20'] = (season_stats.groupby('player_id')[f'{self.source_col}_mean']
                                                    .expanding()
                                                    .var()
                                                    .reset_index(level=0, drop=True)
                                                    ) * self.prior_uncertainty

        df0 = pd.merge(df0,
                       season_stats[
                           list(group_col) +
                           [f'{self.source_col}_mu0',
                            f'{self.source_col}_s2bar',
                            f'{self.source_col}_tau20']
                           ],
                       on=['player_id', 'season'],
                       how='left',
                       )

        return (
            (
                    (
                            df0.n.shift(1) * df0[self.ybar_col] / df0[f'{self.source_col}_s2bar'] +
                            df0[f'{self.source_col}_mu0'] / df0[f'{self.source_col}_tau20']
                    ) /
                    (
                            df0.n.shift(1) / df0[f'{self.source_col}_s2bar'] + 1 / df0[f'{self.source_col}_tau20']
                    )
            ).reset_index(level=0, drop=True)
        )
