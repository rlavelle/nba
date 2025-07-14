from itertools import product
import pickle

from src.models.framework.model import BaseModel


class GridSearch:
    def __init__(self, model: BaseModel, param_grid: dict, metric_fn):
        self.model = model
        self.param_grid = param_grid
        self.metric_fn = metric_fn

    def search(self, X_train, y_train, X_val, y_val):
        best_score = float('inf')
        best_params = None
        best_model = None

        keys, values = zip(*self.param_grid.items())

        for combination in product(*values):
            params = dict(zip(keys, combination))
            model_copy = self._deepcopy_model()

            model_copy.build_model(**params)

            model_copy.train(X_train, y_train)
            score = model_copy.evaluate(X_val, y_val, self.metric_fn)

            print(f"[GRID] Params: {params} => Score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_params = params
                best_model = model_copy

        print(f"[BEST] Params: {best_params} => Score: {best_score:.4f}")
        return best_model, best_params, best_score

    def _deepcopy_model(self):
        return pickle.loads(pickle.dumps(self.model))