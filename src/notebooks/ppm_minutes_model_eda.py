#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 19:10:01 2025

@author: rowanlavelle
"""

"""
There is directional accuracy (~51%) if you use ppm + minutes then std 
the preds indp of the truth, then check for above below mean.
the std is done on the cum rolling mean and std of the ssn
ppm + minutes also gives < 6 MAE over the data

If you take the std data, and then retransform it, but using the real data mu/std
you get a ~9 MAE, but you gain a lot of variance...

%% side node: is there a better way to use hot streak?
    - might make more sense to use hot_streak(1,3,5) against long term averages
    - and also use hot_streak(1,3,5) against avg(1,3,5) (not 1-1)
        
%% ppm logistic regression

%% if logistic reg works, is it possible to use that as signal to predict points O/U mean?
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
from src.feature_engineering.bayes_posterior import BayesPosteriorFeature
from src.feature_engineering.moving_avg import CumSeasonAvgFeature, ExponentialMovingAvgFeature, CumSeasonEMAFeature, SimpleMovingAvgFeature
from src.feature_engineering.base import FeaturePipeline
from src.feature_engineering.player_streak import PlayerHotStreakFeature
from src.feature_engineering.last_game_value import LastGameValueFeature

from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel, RidgeModel, LassoModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel
from src.modeling_framework.framework.model import Model
from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from src.modeling_framework.trainers.rolling_window_trainer import RollingWindowTrainer
from src.modeling_framework.framework.standardizer import Standardizer
from src.modeling_framework.framework.constants import FS_MINUTES, FS_POINTS
from datetime import datetime


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

#%%
x1 = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')
x = x1[x1.points > 0].copy()

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

#%%
ycol = 'ppm'
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
        target='ppm', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date',
        #std=lambda df, fts: Standardizer(idcol='player_id', df=df, features=fts)
    )
)
print(err), preds.mean(), ytest.mean(), preds.var(), ytest.var()
dev_data['lm_preds'] = preds
#dev_data['points_std'] = ytest

yh_train = lm.predict(y_train[FS_POINTS])
preds_proj = Model._proj(y_train[ycol].values, yh_train, preds)
dev_data['lm_preds_proj'] = preds_proj

#%%
dev_data = dev_data.sort_values(by=['player_id', 'season', 'date'])
dev_data['cum_mean_points'] = (
    dev_data.groupby(['player_id', 'season'])['points']
    .expanding()
    .mean()
    .reset_index(drop=True)
    .values
)

#%%
dev_data['preds'] = dev_data.lm_preds * dev_data.minute_pred

#%%
mae(dev_data.preds, dev_data.points)

#%%
dev_data = dev_data.sort_values(by=['player_id', 'season', 'date'])
dev_data['cum_mean_points_s1'] = (
    dev_data.groupby(['player_id', 'season'])['points']
    .expanding()
    .mean()
    .shift(1)
    .reset_index(drop=True)
    .values
)
dev_data['cum_var_points_s1'] = (
    dev_data.groupby(['player_id', 'season'])['points']
    .expanding()
    .std()
    .shift(1)
    .reset_index(drop=True)
    .values
)
dev_data['cum_mean_preds_s1'] = (
    dev_data.groupby(['player_id', 'season'])['preds']
    .expanding()
    .mean()
    .shift(1)
    .reset_index(drop=True)
    .values
)
dev_data['cum_var_preds_s1'] = (
    dev_data.groupby(['player_id', 'season'])['preds']
    .expanding()
    .std()
    .shift(1)
    .reset_index(drop=True)
    .values
)

#%%
z = dev_data.copy()

#%%
z['pred_std'] = (z.preds - z.cum_mean_preds_s1) / (z.cum_var_preds_s1)
z['points_std'] = (z.points - z.cum_mean_points_s1) / (z.cum_var_points_s1)

#%%
#z = dev_data[dev_data.player_slug==np.random.choice(dev_data.player_slug.values)].copy()
#z['pred_std'] = (z.preds - z.cum_mean_preds_s1) / (z.cum_var_preds_s1)
#z['points_std'] = (z.points - z.cum_mean_points_s1) / (z.cum_var_points_s1)
z = z.dropna()

#%%
z['preds2'] = z.pred_std*z.cum_var_points_s1 + z.cum_mean_points_s1

#%%
plt.clf()
plt.scatter(np.arange(z.shape[0]),z.points_std)
plt.scatter(np.arange(z.shape[0]),z.pred_std)
#plt.plot(np.arange(z.shape[0]),z.cum_mean_points, ls='--', color='black')
plt.show()

#%%
a = np.sign(z.points_std)
b = np.sign(z.pred_std)
(a==b).sum() / a.shape[0]

#%%
mae(z.preds2, z.points)

#%%
zz = z[z.player_slug==np.random.choice(z.player_slug.values)].copy()

plt.clf()
plt.scatter(np.arange(zz.shape[0]),zz.points)
plt.scatter(np.arange(zz.shape[0]),zz.preds2)
plt.scatter(np.arange(zz.shape[0]),zz.preds)
plt.scatter(np.arange(zz.shape[0]),zz.lm_preds_proj*zz.minute_pred)
plt.plot(np.arange(zz.shape[0]),zz.cum_mean_points, ls='--', color='black')
plt.show()

#%%
plt.clf()
a = (z.points - z.points.mean()).values
b = (z.preds - z.preds.mean()).values
c = (z.preds2 - z.preds2.mean()).values
a = a[np.argsort(z.points.values)]
b = b[np.argsort(z.points.values)]
c = c[np.argsort(z.points.values)]
plt.clf()
plt.plot(a.cumsum())
plt.plot(b.cumsum())
plt.plot(c.cumsum())
plt.show()

#%%

