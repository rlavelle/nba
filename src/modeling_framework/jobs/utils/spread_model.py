from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_engineering.utils.build_features import get_ft_cols
from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.standardizer import Standardizer
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.standardizers.zscore import ZScoreStandardizer

target = 'spread'


def build_spread_model(train_data: pd.DataFrame) -> Tuple[Model, Standardizer]:
    train_data = train_data.copy()
    train_data[target] = train_data.points.abs()

    params = {
        'learning_rate': 0.01,
        'max_depth': 10,
        'subsample': 0.8,
        'n_estimators': 300,
        'colsample_bytree': 0.5,
        'alpha': 0.05,
        'lambda': 0.05,
    }

    xgb = XGBModel(name='xgb_spread_model')
    xgb.build_model(**params)

    features = get_ft_cols(train_data)

    s = ZScoreStandardizer(idcol=None, features=[target])
    s.fit(train_data)
    # train_data[target] = s.transform(train_data, handle_unseen='global').values

    X_train, y_train = train_data[features], train_data[target]

    xgb.train(X_train, y_train)
    return xgb, s


def predict_spread_model(model: Model,
                         s: ZScoreStandardizer,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> np.array:
    train_data[target] = train_data.points.abs()
    test_data[target] = test_data.points.abs()

    features = get_ft_cols(test_data)

    X_train, y_train = train_data[features], train_data[target]
    X_test = test_data[features]

    preds = model.predict(X_test)
    yh_train = model.predict(X_train)
    proj_preds = Model._proj(y_train, yh_train, preds)

    proj_preds = pd.DataFrame(proj_preds, columns=[target])

    return preds  # s.inverse_transform(proj_preds, handle_unseen='global')

# TODO: rebuild spread model
# def build_spread_results():
#     spread_results = None
#     if spread_model is not None and standardizer is not None and spread_odds is not None:
#         spread_odds = prep_odds(spread_odds, bookmakers=['draftkings'], tomorrow=curr_nxt_date)
#         test_spread_data,_ = format_testing_data(spread_odds, test_game_data)
#
#         spread_preds = None
#         try:
#             spread_preds = predict_spread_model(spread_model,
#                                                 standardizer,
#                                                 train_game_data,
#                                                 test_spread_data)
#             spread_preds = pd.DataFrame(spread_preds, columns=['preds'])
#             #spread_preds.columns = ['preds']
#             spread_preds['team_id'] = test_spread_data.team_id
#
#         except Exception as e:
#             logger.log(f'[ERROR PREDICTING SPREADS]: {e}')
#             insert_error({'msg': str(e)})
#
#         if spread_preds is not None:
#             spread_results = pd.merge(spread_preds, spread_odds, on='team_id', how='left')
