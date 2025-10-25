import numpy as np
import pandas as pd

from src.modeling_framework.framework.constants import FS_MINUTES, FS_POINTS
from src.modeling_framework.framework.model import Model
from src.modeling_framework.models.regression import LinearModel, RidgeModel

target = 'ppm_diff'

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

    minute_model = LinearModel(name='lm_minutes')
    minute_model.build_model()
    minute_model.train(train_data[FS_MINUTES], train_data.minutes)

    test_data['minute_pred'] = minute_model.predict(test_data[FS_MINUTES])

    X_test = test_data[FS_POINTS]
    ppm_diff_preds = model.predict(X_test)
    preds = (test_data.ppm_s1 + ppm_diff_preds) * test_data.minute_pred

    return preds

