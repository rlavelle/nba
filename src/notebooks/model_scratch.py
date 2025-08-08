#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:22:43 2025

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
 'minutes_sma_3'
 ]

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

#%%
x = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')
ycol = 'points'

#%%
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel, RidgeModel, LassoModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel
from src.modeling_framework.framework.model import Model
from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from src.modeling_framework.trainers.rolling_window_trainer import RollingWindowTrainer
from datetime import datetime

#%%
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)

test_data = x[x.date > dev_date_cutoff].copy()
train_data = x[x.date < dev_date_cutoff].copy()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

base_model = SimpleMovingAverageModel(name='ema')
base_model.build_model(window=7, avg_type='ema', source_col=ycol)

# base model
trainer = DateSplitTrainer(model=base_model, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=['points_ema_7'], 
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
params = {
    #'booster': 'gblinear',
    'learning_rate': 0.05, 
    'max_depth': 3, 
    'subsample': 0.6, 
    'n_estimators': 100,
    'colsample_bytree': 0.6,
    'alpha': 10,
}
# params = {'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.6, 'n_estimators': 100}
# params = {'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.6, 'n_estimators': 100}
xgb = XGBModel(name='xgb')
xgb.build_model(**params)

#fs = [f'{c}_ema_{i}' for i in [3,5,7] for c in minutes_best_cols] + [f'{c}_3g_v_{c}_ema_{i}_hot_streak' for i in [5,7] for c in minutes_best_cols]

trainer = DateSplitTrainer(model=xgb, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_POINTS, 
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
lm = RidgeModel(name='lm')
params = {'alpha': 5}
lm.build_model(**params)

trainer = DateSplitTrainer(model=lm, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_POINTS,
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
xgbs = []
lms = []

train_data = train_data.sort_values(by=['player_id', 'season', 'date'])
for k,g in train_data.groupby(train_data.player_id):
    g = g.dropna(subset=FS_POINTS+['points'])
    
    if g.shape[0] == 0:
        continue
    
    if g[g.date >= train_date_cutoff].shape[0] == 0:
        continue
    
    if g[g.date < train_date_cutoff].shape[0] == 0:
        continue
    
    trainer = DateSplitTrainer(model=xgb, metric_fn=mae)
    err, preds, ytest = (
        trainer
        .train_and_evaluate(
            df=g, 
            features=FS_POINTS, 
            target=ycol, 
            proj=False, 
            date_cutoff=train_date_cutoff,
            date_col='date')
    )
        
    xgbs.append(pd.DataFrame({'player_id':k,
                              'points':ytest,
                              'preds':preds}))
    
    trainer = DateSplitTrainer(model=lm, metric_fn=mae)
    err, preds, ytest = (
        trainer
        .train_and_evaluate(
            df=g, 
            features=FS_POINTS,
            target=ycol, 
            proj=False, 
            date_cutoff=train_date_cutoff,
            date_col='date')
    )
    
    lms.append(pd.DataFrame({'player_id':k,
                              'points':ytest,
                              'preds':preds}))

#%%
n = 13
plt.clf()
plt.scatter(xgbs[n].index, xgbs[n].points, alpha=0.5, label='points')
plt.scatter(lms[n].index, lms[n].preds, alpha=0.6, label='lm')
plt.scatter(xgbs[n].index, xgbs[n].preds, alpha=0.6, label='xgb')
plt.hlines([xgbs[n].points.mean()], xmin=0, xmax=len(xgbs[n]), color='black')
plt.legend()
plt.show()

#%%

#%%
np.nanmean(err), preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
from src.modeling_framework.framework.gridsearch import GridSearch
# to beat (minutes)
# EMA ~ 3 np.float64(7.480017809862134) 
# EMA ~ 5 np.float64(7.218518100671039)
# EMA ~ 7 np.float64(7.124904501943216)
# LM np.float64(6.912740297221631)
# LM (fit fwd) np.float64(6.87863321977813)
# [BEST] Params: {'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.6, 'n_estimators': 100} 
# => Score: 6.8875

# to beat (points)
# EMA ~ 7 np.float64(8.158117214164108)
# EMA ~ 5 np.float64(8.3200796117951)
# EMA ~ 3 np.float64(8.709799241654776)
# LM (all ema) np.float64(7.897526694032258)
# LM (ema + hot) np.float64(7.900105179528331)
# LM (fit fwd) np.float64(7.823882642931081)
# [BEST] Params: {'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.6, 'n_estimators': 100} 
# => Score: 7.8573

# LM (fit fwd + minutes model) np.float64(7.82102134851374)

# regression preforms same as xgboost for both minutes and points...

gs = GridSearch(
    model=model,
    metric_fn=rmse,
    param_grid={
        'learning_rate': [0.005, .01, 0.05,],
        'max_depth': [5,10,25,50],
        'subsample': [0.6],
        'n_estimators': [100, 150, 200]
    }
)

m,prm,s = gs.search(
    df=train_data,
    trainer=DateSplitTrainer,
    features=FS_MINUTE,
    target=ycol, 
    proj=False, 
    date_cutoff=train_date_cutoff, 
    date_col='date'
)

#%%
lm = RidgeModel(name='lm')
params = {'alpha': 5}
lm.build_model(**params)

trainer = DateSplitTrainer(model=lm, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=FS_POINTS,
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
lm.model.coef_

#%%
errs = []
ps = []
yt = []
for k, g in train_data.groupby(train_data.player_id):
    trainer = RollingWindowTrainer(model=lm, metric_fn=mae)
    err, preds, ytest = (
        trainer
        .train_and_evaluate(
            df=g, 
            features=FS_POINTS,
            target=ycol, 
            proj=False, 
            window_size=144,
            date_col='date')
    )
    
    errs.append(err)
    ps.extend(preds)
    yt.extend(ytest)
    

#%%
errs = np.array(errs)

#%%
errs.mean(ignore)

#%%
preds = xgb.predict(test_data[FS_POINTS])
preds2 = lm.predict(test_data[FS_POINTS])

#%%
ytest = test_data.points.values

#%%
mae(preds,ytest), mae(preds2, ytest)

#%%
y_train = train_data[train_data.date < train_date_cutoff].copy()
yh_train = m.predict(y_train[FS_POINTS])
yh2_train = lm.predict(y_train[FS_POINTS])

#%%
preds_proj = Model._proj(y_train.points.values, yh_train, preds)
preds2_proj = Model._proj(y_train.points.values, yh2_train, preds2)

#%%
np.corrcoef(preds2, ytest)[0,1]

#%%
yh = preds_proj - preds_proj.mean()
yh2 = preds2_proj - preds2_proj.mean()
yt = ytest - ytest.mean()

xx1 = yh[np.argsort(ytest)]
xx2 = yh2[np.argsort(ytest)]
yy = yt[np.argsort(ytest)]
plt.clf()
plt.plot(np.cumsum(xx1), label='xgb')
plt.plot(np.cumsum(xx2), label='lm')
plt.plot(np.cumsum(yy))
plt.legend()
plt.show()


#%%
plt.clf()
plt.scatter(test_data.index, test_data.points, alpha=0.1)
plt.scatter(test_data.index, preds2, alpha=0.1)
plt.scatter(test_data.index, preds2_proj, alpha=0.1)
plt.show()

