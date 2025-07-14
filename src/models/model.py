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
            self.model = pickle.load(open(os.path.join(self.model_path, self.name), 'rb'))

    def __str__(self):
        return self.name

    @abstractmethod
    def build_model(self):
        pass

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val, metric_fn):
        preds = self.predict(X_val)
        return metric_fn(y_val, preds)

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid: dict, metric_fn):
        best_score = float('inf')
        best_params = None
        best_model = None

        keys, values = zip(*param_grid.items())
        for combination in product(*values):
            params = dict(zip(keys, combination))

            self.build_model(**params)
            self.train(X_train, y_train)
            score = self.evaluate(X_val, y_val, metric_fn)

            print(f"[GRID] Params: {params} => Score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_params = params
                best_model = pickle.loads(pickle.dumps(self.model))  # Deep copy current model

        self.model = best_model
        print(f"[BEST] Params: {best_params} => Score: {best_score:.4f}")
        return best_params, best_score

    def save(self):
        self.model.save(os.path.join(self.model_path, self.name))