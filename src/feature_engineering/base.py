from abc import ABC, abstractmethod
import pandas as pd
from typing import List


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
    def __init__(self, features: List[BaseFeature]):
        self.features = features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all features to the DataFrame"""
        df = df.sort_values(['player_id', 'season', 'date'])
        for feature in self.features:
            df = feature(df)
        return df

    def get_feature_names(self) -> List[str]:
        """Get names of all features in the pipeline"""
        return [f.feature_name for f in self.features]
