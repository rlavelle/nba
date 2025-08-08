#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 14:16:58 2025

@author: rowanlavelle
"""

#%%
# found from forward selection on regression
FS_MINUTE = ['minutes_ema_7',
 'minutes_minutes_cum_ssn_avg_bayes_post',
 'minutes_1g',
 'possessions_3g_v_possessions_ema_5_hot_streak',
 'ppm_1g',
 'estimatedPace_sma_3',
 'estimatedPace_1g',
 'estimatedPace_3g_v_estimatedPace_cum_ssn_avg_hot_streak',
 'pointsOffTurnovers_ema_7',
 'minutes_sma_3',
 'possessions_cum_ssn_avg',
 'oppPointsOffTurnovers_3g_v_oppPointsOffTurnovers_cum_ssn_avg_hot_streak',
 'oppPointsSecondChance_1g',
 'points_1g',
 'possessions_ema_7',
 'possessions_1g',
 'estimatedPace_ema_7',
 'possessions_sma_3',
 'minutes_cum_ssn_avg',
 'freeThrowsPercentage_ema_3',
 'oppPointsSecondChance_sma_3']


FS_POINTS = ['points_points_cum_ssn_ema_bayes_post',
 'fieldGoalsAttempted_ema_5',
 'foulsDrawn_ema_7',
 'fieldGoalsAttempted_fieldGoalsAttempted_cum_ssn_avg_bayes_post',
 'minutes_1g',
 'possessions_cum_ssn_ema',
 'fieldGoalsMade_cum_ssn_avg',
 'fieldGoalsMade_cum_ssn_ema',
 'usagePercentage_ema_3',
 'ppm_cum_ssn_ema',
 'fieldGoalsAttempted_cum_ssn_avg',
 'fieldGoalsMade_fieldGoalsMade_cum_ssn_ema_bayes_post',
 'estimatedUsagePercentage_ema_3',
 'freeThrowsPercentage_ema_3',
 'minutes_cum_ssn_ema',
 'points_cum_ssn_avg',
 'points_cum_ssn_ema',
 'pointsPaint_sma_5',
 'possessions_sma_3',
 'usagePercentage_sma_3',
 'fieldGoalsAttempted_ema_7',
 'minutes_ema_7',
 'possessions_ema_5',
 'percentagePersonalFoulsDrawn_sma_5',
 'usagePercentage_ema_7',
 'PIE_ema_7',
 'minutes_sma_3']

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
from src.feature_engineering.bayes_posterior import BayesPosteriorFeature
from src.feature_engineering.moving_avg import CumSeasonAvgFeature, ExponentialMovingAvgFeature, CumSeasonEMAFeature, SimpleMovingAvgFeature
from src.feature_engineering.base import FeaturePipeline
from src.feature_engineering.player_streak import PlayerHotStreakFeature
from src.feature_engineering.last_game_value import LastGameValueFeature

from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel
from src.modeling_framework.framework.model import Model
from src.modeling_framework.framework.gridsearch import GridSearch

from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from datetime import datetime

#%%
x = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')

#%%
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

#%%
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)
test_data = x[x.date > dev_date_cutoff].copy()
train_data = x[x.date < dev_date_cutoff].copy()

#%%
minute_model = LinearModel(name='lm_minutes')
minute_model.build_model()

trainer = DateSplitTrainer(model=minute_model, metric_fn=rmse)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_MINUTE,
        target='minutes', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)

#%%
x['minute_pred'] = minute_model.predict(x[FS_MINUTE])

#%%
points_model = LinearModel(name='lm_points')
points_model.build_model()

test_data = x[x.date > dev_date_cutoff].copy()
dev_data = x[(x.date < dev_date_cutoff) & (x.date > train_date_cutoff)].copy()
train_data = x[x.date < dev_date_cutoff].copy()

trainer = DateSplitTrainer(model=points_model, metric_fn=rmse)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_POINTS + ['minute_pred'],
        target='points', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)

#%%
test_data = x[x.date > dev_date_cutoff].copy()
test_data = test_data.sort_values(by='date')
ytest = test_data.points.values
preds = points_model.predict(test_data[FS_POINTS + ['minute_pred']])
rmse(preds, ytest)
err = np.sqrt((preds-ytest)**2)

# test_data = dev_data.sort_values(by='date')
# ytest = dev_data.points.values
# preds = points_model.predict(dev_data[FS_POINTS + ['minute_pred']])
# rmse(preds, ytest)

tmp = test_data.copy()
tmp['preds'] = preds
tmp['ngame'] = tmp.groupby('player_id').game_id.cumcount() + 1

y_train = train_data[train_data.date < train_date_cutoff].copy()
yh_train = points_model.predict(y_train[FS_POINTS + ['minute_pred']])
preds_proj = Model._proj(y_train.points.values, yh_train, preds)

tmp['preds_proj'] = preds_proj
tmp['err'] = err
tmp['mu_err'] = err.mean()
tmp['proj_err'] = np.sqrt((tmp.preds_proj - tmp.points)**2)

#%%
"""
At the very end of the season there is a large spike in error....
most likely due to teams trying to make the playoffs?

does not happen in the dev set, could be random, or overfitting?

up to ~4000 points into the test data we have a pretty good random walk
there seems to be no systematic miss prediction in early season (thanks to bayes?)
"""
plt.clf()
pp = preds - preds.mean()
yy = ytest - ytest.mean()
plt.plot((err - err.mean()).cumsum())
plt.show()

#%%
plt.clf()
z1 = tmp.groupby(tmp.ngame).err.mean()
plt.plot(z1)
plt.show()

#%%
z = tmp[tmp.player_slug == 'lebron-james']
z.shape

#%%
z.err.sum(), z.proj_err.sum()

#%%
plt.clf()
plt.plot(z.points)
plt.plot(z.preds)
plt.plot(z.preds_proj)
plt.show()


