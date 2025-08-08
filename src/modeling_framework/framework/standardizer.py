from typing import List
import pandas as pd


class Standardizer:
    def __init__(self, idcol: str, df: pd.DataFrame, features: List[str]):
        self.idcol = idcol
        self.features = features

        self._compute_stats(df)

    def _compute_stats(self, df: pd.DataFrame):
        grouped = df.groupby(self.idcol)[self.features]
        self.mu = grouped.mean()
        self.std = grouped.std()
        self.std = self.std.mask(self.std == 0, 1e-8)

    def transform(self, df: pd.DataFrame, handle_unseen: str = 'error') -> pd.DataFrame:
        """
        :param df: input dataframe to be standardized
        :param handle_unseen: How to handle IDs not seen during fitting.
            'error' - raise ValueError (default)
            'nan' - return NaN for those rows
            'global' - use global mean/std instead
        :return: standardized features
        """
        if self.idcol not in df:
            raise ValueError(f'{self.idcol} not in provided dataframe')

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        dff = df.set_index(self.idcol)
        unseen_ids = dff.index.difference(self.mu.index)

        if not unseen_ids.empty:
            if handle_unseen == 'error':
                raise ValueError(f"Found unseen IDs: {unseen_ids.tolist()}")
            elif handle_unseen == 'global':
                global_mu = self.mu.mean()
                global_std = self.std.mean()

                mu_filled = self.mu.copy()
                std_filled = self.std.copy()

                for player_id in unseen_ids:
                    mu_filled.loc[player_id] = global_mu
                    std_filled.loc[player_id] = global_std

                mu_filled = mu_filled.reindex(dff.index)
                std_filled = std_filled.reindex(dff.index)

                standardized = (dff[self.features] - mu_filled) / std_filled

            elif handle_unseen == 'nan':
                standardized = (dff[self.features] - self.mu) / self.std
            else:
                raise ValueError(f"Invalid handle_unseen: {handle_unseen}")
        else:
            standardized = (dff[self.features] - self.mu) / self.std

        return standardized.reset_index()