import numpy as np
import pandas as pd

from src.db.schema import PLAYER_PROP_RESULTS_SCHEMA
from src.db.utils import insert_table, insert_error
from src.logging.logger import Logger
from src.modeling_framework.framework.constants import FS_MINUTES, FS_POINTS
from src.modeling_framework.framework.model import Model
from src.modeling_framework.jobs.utils.formatting import prep_odds
from src.modeling_framework.models.regression import LinearModel, RidgeModel

target = 'ppm_diff'


def insert_prop_results(prop_results):
    cols = [
        'player_id', 'dint', 'bookmaker', 'odd_type', 'description',
        'price', 'point', 'preds'
    ]

    insert_table(prop_results[cols], PLAYER_PROP_RESULTS_SCHEMA, 'player_prop_results')


def build_player_prop_model(train_data:pd.DataFrame) -> Model:
    X_train, y_train = train_data[FS_POINTS], train_data[target]

    params = {'alpha': 5}
    lm = RidgeModel(name='player_ppm_diff_model')
    lm.build_model(**params)
    lm.train(X_train, y_train)

    return lm

def predict_player_prop_model(model:Model,
                              train_data:pd.DataFrame,
                              test_data:pd.DataFrame) -> np.array:

    test_data = test_data.copy()
    minute_model = LinearModel(name='lm_minutes')
    minute_model.build_model()
    minute_model.train(train_data[FS_MINUTES], train_data.minutes)

    test_data['minute_pred'] = minute_model.predict(test_data[FS_MINUTES])

    X_test = test_data[FS_POINTS]
    ppm_diff_preds = model.predict(X_test)
    preds = (test_data.ppm_s1 + ppm_diff_preds) * test_data.minute_pred

    return preds


def build_player_prop_results(prop_model: Model,
                              prop_odds: pd.DataFrame,
                              curr_date: int,
                              train_player_data: pd.DataFrame,
                              test_player_data: pd.DataFrame,
                              logger: Logger):
    prop_results = None
    if prop_model is not None and prop_odds is not None:
        prop_preds = None
        try:
            prop_preds = predict_player_prop_model(prop_model, train_player_data, test_player_data)
            prop_preds = pd.DataFrame(prop_preds, columns=['preds'])
            prop_preds['player_id'] = test_player_data.player_id

        except Exception as e:
            logger.log(f'[ERROR PREDICTING PLAYER PROPS]: {e}')
            insert_error({'msg': str(e)})

        if prop_preds is not None:
            # TODO: build this out to all?
            prop_odds = prep_odds(prop_odds, bookmakers=['draftkings'], curr_date=curr_date)
            prop_results = pd.merge(prop_preds, prop_odds, on='player_id', how='left')
            logger.log(f'[MISSING PLAYERS FROM MODEL]: {prop_odds.price.isna().sum()}')

    return prop_results