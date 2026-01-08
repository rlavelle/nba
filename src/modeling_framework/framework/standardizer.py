from abc import ABC
from typing import List

import pandas as pd


class Standardizer(ABC):
    def __init__(self, idcol: str, features: List[str]):
        self.idcol = idcol
        self.features = features

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, handle_unseen: str = 'error') -> pd.DataFrame:
        pass

    def inverse_transform(self, df: pd.DataFrame, handle_unseen: str = 'error'):
        pass
