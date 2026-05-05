import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd

from src.logging.logger import Logger


class BaseFeature(ABC):
    """Abstract base class for all feature engineering operations"""

    def __init__(self, group_col: tuple[str]):
        """
        Initialize base feature with grouping columns

        Args:
            group_col: Tuple of column names to group by (e.g., ('player_id', 'season'))
        """
        self.group_col = group_col

    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Name of the generated feature column"""
        pass

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate feature values for the entire DataFrame"""
        pass

    def calculate(self, df: pd.DataFrame, date_col: str = 'date') -> pd.Series:
        """
        Calculate feature values for the entire DataFrame with sorting guarantee.

        Args:
            df: Input DataFrame
            date_col: Time column name for ordering within groups (default: 'date')

        Returns:
            Calculated feature as Series aligned with original index
        """
        df_work = df.copy()

        sort_cols = list(self.group_col)
        if date_col in df_work.columns and date_col not in sort_cols:
            sort_cols.append(date_col)

        df_sorted = df_work.sort_values(by=sort_cols).reset_index(drop=True)

        return self._calculate(df_sorted)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to apply the feature and return updated DataFrame"""
        df[self.feature_name] = self._calculate(df).values
        return df


class FeaturePipeline:
    def __init__(self, features: List[BaseFeature], logger: Logger = None):
        self.features = features
        self.logger = logger

    def transform(self,
                  df: pd.DataFrame,
                  sort_order: tuple[str] = ('player_id', 'season', 'date'),
                  max_workers: int = 4,
                  use_multi_processing: bool = False) -> pd.DataFrame:
        """Apply all features to the DataFrame in parallel batches"""
        df = df.sort_values(list(sort_order))

        if use_multi_processing:
            new_columns = self._transform_multi_processing(df, max_workers)
        else:
            new_columns = self._transform_batch(df)

        return self._concat_columns(df, new_columns)

    def _transform_multi_processing(self, df, max_workers):
        new_columns = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all feature calculations
            future_to_feature = {
                executor.submit(self._safe_calculate, feature, df): feature
                for feature in self.features
            }

            # Collect results as they complete
            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    feature_name, result = future.result()
                    if result is not None:
                        new_columns[feature_name] = result
                except Exception as e:
                    if self.logger:
                        self.logger.log(f"[ERROR]: {feature.feature_name}: {e}")

        return new_columns

    def _transform_batch(self, df: pd.DataFrame):
        new_columns = {}
        for feature in self.features:
            try:
                new_columns[feature.feature_name] = feature.calculate(df)
            except Exception as e:
                if self.logger:
                    self.logger.log(f"[ERROR]: {feature.feature_name}: {e}")

        return new_columns

    def _safe_calculate(self, feature, df):
        """Wrapper for safe feature calculation"""
        try:
            result = feature.calculate(df)
            return feature.feature_name, result
        except Exception as e:
            return feature.feature_name, None

    def _concat_columns(self, df, new_columns):
        try:
            # TODO: what is this mess hahahahaha... someway only way the concat wouldnt fail
            z = np.array([a.values for a in new_columns.values()])
            dfn = pd.DataFrame(data=z.T, columns=list(new_columns.keys()), index=df.index)
            df = pd.concat([df, dfn], axis=1)
            return df

        except Exception as e:
            if self.logger:
                self.logger.log(f'[ERROR ON DF CONCAT]: {e}')

    def get_feature_names(self) -> List[str]:
        """Get names of all features in the pipeline"""
        return [f.feature_name for f in self.features]
