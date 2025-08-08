from src.modeling_framework.framework.model import Model


class LastGameModel(Model):
    def build_model(self, **params):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        return X


class SimpleMovingAverageModel(Model):
    def build_model(self, **params):
        self.window = params['window']
        self.avg_type = params['avg_type']
        self.source_col = params['source_col']

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        col_name = f'{self.source_col}_{self.avg_type}_{self.window}'
        return X[col_name].values
