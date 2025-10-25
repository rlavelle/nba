import numpy as np
import pandas as pd

from src.feature_engineering.utils.build_features import get_ft_cols
from src.modeling_framework.framework.model import Model
from src.modeling_framework.models.xgb_model import XGBModel

target = 'home_win'

def build_money_line_model(train_data:pd.DataFrame) -> Model:
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

def predict_money_line_model(model:Model,
                             test_data:pd.DataFrame) -> np.array:

    test_data[target] = (test_data.points > 0).astype(int)
    features = get_ft_cols(test_data)
    X_test = test_data[features]

    return model.predict(X_test)