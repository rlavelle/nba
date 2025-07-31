from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.modeling_framework.framework.model import Model


class LinearModel(Model):
    def build_model(self, **params):
        self.model = LinearRegression()

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class RidgeModel(Model):
    def build_model(self, **params):
        self.model = Ridge(**params)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class LassoModel(Model):
    def build_model(self, **params):
        self.model = Lasso(**params)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
