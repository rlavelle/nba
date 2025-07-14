import pickle
from xgboost import XGBRegressor
from src.models.framework.model import Model
from src.models.framework.model import BaseModel


class XGBModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = XGBRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

class XGBWrapper(BaseModel):
    def build_model(self, **kwargs):
        self.model = XGBModel(**kwargs)