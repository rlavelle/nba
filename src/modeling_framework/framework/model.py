import configparser
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.config import CONFIG_PATH


class Model(ABC):
    def __init__(self, name, load=False):
        self.name = name
        self.model: Any = None

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        self.model_path = config.get('MODEL_PATH', 'path')

        if load:
            assert os.path.exists(
                os.path.join(self.model_path, name)), f'{os.path.join(self.model_path, name)} does not exist'
            self.load()

    def __str__(self):
        return self.name

    def train(self, X_train, y_train):
        self.fit(X_train, y_train)

    def evaluate(self, X_val, y_val, metric_fn):
        preds = self.predict(X_val)
        return metric_fn(y_val, preds)

    def stats(self, X, y_true):
        return {
            'rmse': np.sqrt(np.mean(np.pow(y_true - self.predict(X), 2))),
            'game_corr': np.corrcoef(y_true, self.predict(X))[0,1]
        }

    def load(self, fpath=None):
        pth = fpath if fpath else os.path.join(self.model_path, self.name)
        self.model = pickle.load(open(pth, 'rb'))

    def save(self, fpath=None):
        pth = fpath if fpath else os.path.join(self.model_path, self.name)
        pickle.dump(self, open(pth , 'wb'))

    @staticmethod
    def _proj(y_train, yh_train, yh_test):
        assert y_train.shape == yh_train.shape, 'train truth and hat diff length'
        yt = y_train - y_train.mean()
        yh = yh_train - yh_train.mean()

        b = (yh.T @ yt) / (yh.T @ yh)
        yc = y_train.mean() + b * (yh_test-yh_train.mean())
        yf = y_train.mean() + (y_train.std() / yc.std()) * (yc - y_train.mean())
        return yf

    @abstractmethod
    def build_model(self, **params):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X_train, Y_train):
        pass
