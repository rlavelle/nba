from itertools import product
import pickle
from typing import Dict, Any, Tuple, List
import pandas as pd

from src.logging.logger import Logger
from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.trainer import Trainer


class GridSearch:
    def __init__(self,
                 model: Model,
                 trainer: Trainer,
                 param_grid: Dict[str, Any],
                 logger: Logger = None):
        self.model = model
        self.trainer = trainer
        self.param_grid = param_grid
        self.logger = logger

    def search(self,
               df: pd.DataFrame,
               features: List[str],
               target: str,
               **trainer_kwargs) -> Tuple[Model, Dict, float]:
        """
        Generic grid search that works with any Trainer implementation

        Args:
            df: DataFrame containing all data
            features: List of feature columns
            target: Target column name
            trainer_kwargs: Arguments specific to the trainer implementation

        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        best_score = float('inf')
        best_params = None
        best_model = None

        keys, values = zip(*self.param_grid.items())

        for combination in product(*values):
            params = dict(zip(keys, combination))
            self.model.build_model(**params)

            # Delegate all evaluation to the trainer
            error, _, _ = self.trainer.train_and_evaluate(
                df=df,
                features=features,
                target=target,
                **trainer_kwargs
            )

            score = self._aggregate_errors(error)

            if score < best_score:
                msg = f"[GRID] Params: {params} => Score: {score:.4f}"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)

                best_score = score
                best_params = params
                best_model = self._deepcopy_model()


        msg = f"[BEST] Params: {best_params} => Score: {score:.4f}"
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

        return best_model, best_params, best_score

    def _aggregate_errors(self, errors: float) -> float:
        """Handle different error return formats from trainers"""
        if not errors:
            return float('inf')
        if isinstance(errors, (float, int)):  # For trainers returning single value
            return errors
        return float('inf')

    def _deepcopy_model(self):
        return pickle.loads(pickle.dumps(self.model))