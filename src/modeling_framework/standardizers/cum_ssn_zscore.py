import pandas as pd

from src.modeling_framework.framework.standardizer import Standardizer


class CumSsnZScoreStandardizer(Standardizer):
    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, handle_unseen: str = 'error') -> pd.DataFrame:
        """
        :param df: input dataframe to be standardized
        :param handle_unseen:
        :return: standardized features
        """
        if self.idcol not in df:
            raise ValueError(f'{self.idcol} not in provided dataframe')

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        dff = df.copy()
        dff = dff.sort_values(by=['player_id', 'season', 'date'])
        mus = (
            dff.groupby([dff.player_id, dff.season])[self.features]
            .expanding()
            .mean()
            .shift(1)
            .set_index(df.index)
        )

        # use cum player points std
        stds = (
            dff.groupby([dff.player_id])[self.features]
            .expanding()
            .std()
            .shift(1)
            .set_index(df.index)
        )

        stds = stds.mask(stds == 0, 1e+8)

        standardized = (df[self.features] - mus) / stds
        return standardized.fillna(0).reset_index()

    def inverse_transform(self, df: pd.DataFrame, handle_unseen: str = 'error'):
        """
               :param df: input dataframe to be inverse transformed
               :param handle_unseen:
               :return: unstandardized features
               """
        if self.idcol not in df:
            raise ValueError(f'{self.idcol} not in provided dataframe')

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        dff = df.copy()
        dff = dff.sort_values(by=['player_id', 'season', 'date'])
        mus = (
            dff.groupby([dff.player_id, dff.season])[self.features]
            .expanding()
            .mean()
            .shift(1)
        )
        stds = (
            dff.groupby([dff.player_id, dff.season])[self.features]
            .expanding()
            .std()
            .shift(1)
        )

        standardized = df[self.features] * stds + mus
        return standardized.reset_index()
