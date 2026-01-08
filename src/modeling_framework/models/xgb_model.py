from xgboost import XGBRegressor

from src.modeling_framework.framework.model import Model


class XGBModel(Model):
    def build_model(self, **params):
        self.model = XGBRegressor(**params)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
