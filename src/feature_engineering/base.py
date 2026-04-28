import pickle
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd

from src.logging.logger import Logger


class BaseFeature(ABC):
    """Abstract base class for all feature engineering operations"""

    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Name of the generated feature column"""
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate feature values for the entire DataFrame"""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to apply the feature and return updated DataFrame"""
        df[self.feature_name] = self.calculate(df).values
        return df


class FeaturePipeline:
    def __init__(self, features: List[BaseFeature], logger: Logger = None):
        self.features = features
        self.logger = logger

    def transform(self,
                  df: pd.DataFrame,
                  sort_order: tuple[str] = ('player_id', 'season', 'date')) -> pd.DataFrame:
        """Apply all features to the DataFrame in batches"""
        df = df.sort_values(list(sort_order))

        new_columns = {}
        for feature in self.features:
            try:
                new_columns[feature.feature_name] = feature.calculate(df)
            except Exception as e:
                if self.logger:
                    self.logger.log(f"[ERROR]: {feature.feature_name}: {e}")

        if new_columns:
            try:
                z = np.array([a.values for a in new_columns.values()])
                dfn = pd.DataFrame(data=z.T, columns=list(new_columns.keys()), index=df.index)
                df = pd.concat([df, dfn], axis=1)

            except Exception as e:
                if self.logger:
                    self.logger.log(f'[ERROR ON DF CONCAT]: {e}')

        return df

    def get_feature_names(self) -> List[str]:
        """Get names of all features in the pipeline"""
        return [f.feature_name for f in self.features]
