#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:38:07 2025

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
players = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')
games = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/game_feature_df.pckl')

#%%
opponents = pd.merge(
    games[["game_id", "team_id"]],
    games[["game_id", "team_id"]],
    on="game_id"
)
opponents = opponents[opponents["team_id_x"] != opponents["team_id_y"]]
opponents = opponents.rename(
    columns={"team_id_x": "team_id", "team_id_y": "opp_team_id"}
)

players = pd.merge(
    players,
    opponents,
    on=["game_id", "team_id"]
)

players = pd.merge(
    players,
    games,
    left_on=["game_id", "opp_team_id"],
    right_on=["game_id", "team_id"],
    suffixes=('', '_opp')
)

#%%
players = players[players.points > 0].copy()

#%%
COLS = list(set([
        'game_id', 'team_id', 'player_id', 'season', 'date', 'player_slug',
        'points', 'minutes', 'ppm'
] + FS_MINUTES + FS_POINTS + [c for c in players.columns if c.endswith('_opp')]))

#%%
x = players[COLS].copy()

#%%
x = x.sort_values(by=['player_id', 'season', 'date'])
x['tmp'] = (
    x.groupby(['player_id', 'season'])['points']
    .expanding()
    .mean()
    .shift(1)
    .reset_index(drop=True)
    .values
)
x['points_o_u'] = (x.points > x.tmp).astype(int)


x['ppm_s1'] = (
    x.groupby(['player_id', 'season'])['ppm']
    .expanding()
    .mean()
    .shift(1)
    .fillna(0)
    .reset_index(drop=True)
    .values
)
x['ppm_o_u'] = (x.ppm > x.ppm_s1).astype(int)
x['ppm_diff'] = x.ppm - x.ppm_s1
x = x.drop(columns=['tmp'])

#%%
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)

train_data = x[x.date < dev_date_cutoff].copy()

minute_model = LinearModel(name='lm_minutes')
minute_model.build_model()

trainer = DateSplitTrainer(model=minute_model, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_MINUTES,
        target='minutes', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)

x['minute_pred'] = minute_model.predict(x[FS_MINUTES])
mae(x.minute_pred, x.minutes)

#%%
ycol = 'ppm_diff'
test_data = x[x.date > dev_date_cutoff].copy()
dev_data = x[(x.date > train_date_cutoff) & (x.date < dev_date_cutoff)].copy()
y_train = x[x.date < train_date_cutoff].copy()
train_data = x[x.date < dev_date_cutoff].copy()

#%%
lm = RidgeModel(name='lm')
params = {'alpha': 5}
lm.build_model(**params)

#%%
trainer = DateSplitTrainer(model=lm, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_POINTS,
        target='ppm_diff', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date',
    )
)
print(err), preds.mean(), ytest.mean(), preds.var(), ytest.var()
dev_data['lm_preds'] = preds

dev_data['preds'] = (dev_data.ppm_s1 + dev_data.lm_preds) * dev_data.minute_pred
mae(dev_data.preds, dev_data.points)

#%%
def build_quick_lm(fts):
    trainer = DateSplitTrainer(model=lm, metric_fn=rmse)
    err, preds, ytest = (
        trainer
        .train_and_evaluate(
            df=train_data, 
            features=fts, 
            target=ycol, 
            proj=False, 
            date_cutoff=train_date_cutoff,
            date_col='date')
    )
    preds = (dev_data.ppm_s1 + preds) * dev_data.minute_pred
    #err, preds.mean(), ytest.mean(), preds.var(), ytest.var()
    return mae(preds, dev_data.points)

def bic(model, X, y):
    yh = (dev_data.ppm_s1 + model.predict(X)) * dev_data.minute_pred
    rss = np.sum((y-yh)**2)
    n = len(y)
    k = X.shape[1]+1
    return n * np.log(rss/n) + k * np.log(n)

#%%
wanted_subs = ["_bayes_post", "_1g", "_sma_", "_ema_", "_cum_ssn_", "_hot_streak"]

cols = [
    col for col in players.columns
    if col.endswith("_opp") and 'defensiveRating' in col and any(sub in col for sub in wanted_subs)
]

curr_order = [(FS_POINTS, np.inf), ]

#%%
prev_bic = bic(lm, dev_data[FS_POINTS], dev_data.points)
prev_bic

#%%
while len(cols) > 0:
    best_col = None
    best_err = np.inf
    for col in cols:
        fts = curr_order[-1][0] + [col]
        e = build_quick_lm(fts)
        if e < best_err:
            best_col = col
            best_err = e
            bic_v = bic(lm, dev_data[fts], dev_data[ycol])
    
    print(bic_v - prev_bic)
    prev_bic = bic_v
    curr_order.append((
        curr_order[-1][0] + [best_col], best_err
    ))
    cols.remove(best_col)
    print(len(cols), curr_order[-1][1])
    
#%%
curr_order
