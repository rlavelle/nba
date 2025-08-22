#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:30:15 2025

@author: rowanlavelle
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib auto

from src.db.db_manager import DBManager
from src.modeling_framework.framework.dataloader import NBADataLoader
from src.types.player_types import PlayerType, PLAYER_FEATURES
from src.types.game_types import GAME_FEATURES
from src.feature_engineering.bayes_posterior import BayesPosteriorFeature
from src.feature_engineering.moving_avg import CumSeasonAvgFeature, ExponentialMovingAvgFeature, CumSeasonEMAFeature, SimpleMovingAvgFeature
from src.feature_engineering.base import FeaturePipeline
from src.feature_engineering.player_streak import PlayerHotStreakFeature
from src.feature_engineering.last_game_value import LastGameValueFeature

from src.db.db_manager import DBManager
from src.modeling_framework.framework.dataloader import NBADataLoader
from src.types.player_types import PlayerType, PLAYER_FEATURES
from src.feature_engineering.bayes_posterior import BayesPosteriorFeature
from src.feature_engineering.moving_avg import CumSeasonAvgFeature, ExponentialMovingAvgFeature, CumSeasonEMAFeature, SimpleMovingAvgFeature
from src.feature_engineering.base import FeaturePipeline
from src.feature_engineering.player_streak import PlayerHotStreakFeature
from src.feature_engineering.last_game_value import LastGameValueFeature

from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel, RidgeModel, LassoModel, LogitModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel
from src.modeling_framework.framework.model import Model
from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from src.modeling_framework.trainers.rolling_window_trainer import RollingWindowTrainer
from src.modeling_framework.standardizers.zscore import ZScoreStandardizer
from src.modeling_framework.standardizers.cum_ssn_zscore import CumSsnZScoreStandardizer
from src.modeling_framework.framework.constants import FS_MINUTES, FS_POINTS
from datetime import datetime


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

#%%
x = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/game_feature_df.pckl')

meta_cols = [
    "season", "season_type", "season_type_code", "dint",
    "date", "team_id", "is_home", "stat_type"
]

home = x[x["is_home"] == 1].set_index("game_id")
away = x[x["is_home"] == 0].set_index("game_id")
meta = home[meta_cols].reset_index()

home_stats = home.drop(columns=meta_cols)
away_stats = away.drop(columns=meta_cols)

diff_stats = home_stats - away_stats
diff_stats = diff_stats.reset_index()

df_diff = pd.concat([meta, diff_stats.drop(columns=["game_id"])], axis=1)
df_diff['home_win'] = (df_diff.points > 0).astype(int)

#%%
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel

from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from datetime import datetime
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)
test_data = df_diff[df_diff.date > dev_date_cutoff].copy()
train_data = df_diff[df_diff.date < dev_date_cutoff].copy()

#%%

#%%
params = {
    'learning_rate': 0.01, 
    'max_depth': 5, 
    'subsample': 1, 
    'n_estimators': 300,
    'colsample_bytree': 0.5,
    'alpha': 10,
}

xgb = XGBModel(name='xgb')
xgb.build_model(**params)

#%%
trainer = DateSplitTrainer(model=xgb, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=cols,
        target='home_win', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date',
    )
)

#%%
a = ytest - ytest.mean()
b = preds - preds.mean()
aa = a[np.argsort(ytest)]
bb = b[np.argsort(ytest)]
plt.clf()
#plt.plot(aa.cumsum())
plt.plot(bb.cumsum())
plt.show()

#%%
np.quantile(preds, q=[0.25,0.5,0.75,1])

#%%
p = []
pp = []
for i in np.arange(0,0.85,0.05):
  mh = preds[preds > i]
  m = ytest[preds > i]
  p.append(m.mean())
  pp.append(i)
  print(m.mean(), i)
  
plt.clf()
plt.plot(pp,p)
plt.show()

#%%

  



