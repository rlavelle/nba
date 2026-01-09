import numpy as np
import pandas as pd

from src.db.schema import GAME_ODDS_ML_RESULTS_SCHEMA
from src.db.utils import insert_table, insert_error
from src.feature_engineering.utils.build_features import get_ft_cols
from src.logging.logger import Logger
from src.modeling_framework.framework.model import Model
from src.modeling_framework.jobs.utils.formatting import prep_odds, format_testing_data
from src.modeling_framework.models.xgb_model import XGBModel

target = 'home_win'


def insert_ml_results(ml_results: pd.DataFrame):
    cols = [
        'team_id', 'dint', 'bookmaker', 'price', 'preds'
    ]

    insert_table(ml_results[cols], GAME_ODDS_ML_RESULTS_SCHEMA, 'game_ml_results')


def build_money_line_model(train_data: pd.DataFrame) -> Model:
    train_data = train_data.copy()
    train_data[target] = (train_data.points > 0).astype(int)

    params = {
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 1,
        'n_estimators': 300,
        'colsample_bytree': 0.5,
        'alpha': 10,
    }

    xgb = XGBModel(name='xgb_ml_model')
    xgb.build_model(**params)

    features = get_ft_cols(train_data)
    X_train, y_train = train_data[features], train_data[target]

    xgb.train(X_train, y_train)
    return xgb


def predict_money_line_model(model: Model,
                             test_data: pd.DataFrame) -> np.array:
    features = get_ft_cols(test_data)
    X_test = test_data[features]

    return model.predict(X_test)


def build_money_line_results(money_line_model: Model,
                             money_line_odds: pd.DataFrame,
                             curr_date: int,
                             test_game_data: pd.DataFrame,
                             logger: Logger):
    ml_results = None
    if money_line_model is not None and money_line_odds is not None:
        money_line_odds = prep_odds(money_line_odds, bookmakers=['draftkings'], curr_date=curr_date)
        test_ml_data_home, test_ml_data_away = format_testing_data(money_line_odds, test_game_data)

        ml_preds_home = None
        ml_preds_away = None
        try:
            # NOTE: in theory, the ML model is trained on diff to predict win / loss
            #       so if its generalized well it should be able to predict the flip...
            ml_preds_home = predict_money_line_model(money_line_model,
                                                     test_ml_data_home)
            ml_preds_away = predict_money_line_model(money_line_model,
                                                     test_ml_data_away)

            ml_preds_home = pd.DataFrame(ml_preds_home, columns=['preds'])
            ml_preds_away = pd.DataFrame(ml_preds_away, columns=['preds'])

            ml_preds_home['team_id'] = test_ml_data_home.team_id
            ml_preds_away['team_id'] = test_ml_data_away.team_id

        except Exception as e:
            logger.log(f'[ERROR PREDICTING MONEYLINE]: {e}')
            insert_error({'msg': str(e)})

        if ml_preds_home is not None and ml_preds_away is not None:
            ml_results1 = pd.merge(ml_preds_home, money_line_odds, on='team_id', how='left')
            ml_results2 = pd.merge(ml_preds_away, money_line_odds, on='team_id', how='left')
            ml_results = pd.concat([ml_results1, ml_results2]).sort_values(by=['game_id', 'is_home'])

    return ml_results
