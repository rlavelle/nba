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

        if params['source_col']:
            self.source_col = params['source_col']
        else:
            self.source_col = 'points'

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        col_name = f'{self.source_col}_sma_{self.window}'
        return X[col_name].values
