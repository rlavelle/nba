from typing import List
import pandas as pd

from src.modeling_framework.framework.standardizer import Standardizer


class ZScoreStandardizer(Standardizer):
    def fit(self, df: pd.DataFrame):
        if self.idcol:
            grouped = df.groupby(self.idcol)[self.features]
        else:
            grouped = df[self.features].copy()

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
        if self.idcol and self.idcol not in df:
            raise ValueError(f'{self.idcol} not in provided dataframe')

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        if self.idcol:
            dff = df.set_index(self.idcol)
            unseen_ids = dff.index.difference(self.mu.index)
        else:
            dff = df.copy()
            unseen_ids = pd.DataFrame()

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

        return standardized.reset_index(drop=True)

    def inverse_transform(self, df: pd.DataFrame, handle_unseen: str = 'error'):
        """
               :param df: input dataframe to be inverse transformed
               :param handle_unseen: How to handle IDs not seen during fitting.
                   'error' - raise ValueError (default)
                   'nan' - return NaN for those rows
                   'global' - use global mean/std instead
               :return: unstandardized features
               """
        if self.idcol and self.idcol not in df:
            raise ValueError(f'{self.idcol} not in provided dataframe')

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        if self.idcol:
            dff = df.set_index(self.idcol)
            unseen_ids = dff.index.difference(self.mu.index)
        else:
            dff = df.copy()
            unseen_ids = pd.DataFrame()

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

                standardized = dff[self.features]*std_filled + mu_filled

            elif handle_unseen == 'nan':
                standardized = dff[self.features]*self.std + self.mu
            else:
                raise ValueError(f"Invalid handle_unseen: {handle_unseen}")
        else:
            standardized = dff[self.features]*self.std + self.mu

        return standardized.reset_index(drop=True)
