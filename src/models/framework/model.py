import configparser
import os
import pickle
from abc import ABC, abstractmethod
from itertools import product

from src.config import CONFIG_PATH


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X_train, Y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, fpath):
        pickle.dump(self, open(fpath, 'wb'))


class BaseModel(ABC):
    def __init__(self, name: str, load=False):
        self.name = name
        self.model:Model = None

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        self.model_path = config.get('MODEL_PATH', 'path')

        if load:
            assert os.path.exists(os.path.join(self.model_path, name)), f'{os.path.join(self.model_path, name)} does not exist'
            self.load()

    def __str__(self):
        return self.name

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val, metric_fn):
        preds = self.predict(X_val)
        return metric_fn(y_val, preds)

    def save(self):
        self.model.save(os.path.join(self.model_path, self.name))

    def load(self):
        self.model = pickle.load(open(os.path.join(self.model_path, self.name), 'rb'))
