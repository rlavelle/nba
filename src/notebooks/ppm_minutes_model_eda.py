#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 19:10:01 2025

@author: rowanlavelle
"""

"""
%% reasonable model (if chasing variance use projected values)
    - model = [cum ssn mu + hat(ppm_diff)]*hat(minutes)
    - proj:     MAE ~6.7 | err std: ~8.8 | var: 57
    - non-proj: MAE ~5.6 | err std: ~7.2 | var: 31
    
    - model = 5g sma
    -           MAE ~6.1 | err std: ~7.9 | var: 43
    
    - model = cum ssn ema
    -           MAE ~5.8 | err std: ~7.5 | var: 34
    
There is directional accuracy (~51%) if you use ppm + minutes then std (inside random chance)
the preds indp of the truth, then check for above below mean.
the std is done on the cum rolling mean and std of the ssn
ppm + minutes also gives < 6 MAE over the data

If you take the std data, and then retransform it, but using the real data mu/std
you get a ~9 MAE, but you gain a lot of variance...

%% side node: is there a better way to use hot streak?
    - might make more sense to use hot_streak(1,3,5) against long term averages
    - and also use hot_streak(1,3,5) against avg(1,3,5) (not 1-1)
        
%% ppm logistic regression
    - works OOS ~55% accuracy (outside random chance)
    - model does not fit all that well 

%% if logistic reg works, is it possible to use that as signal to predict points O/U mean?
    - model that predicts PPM O/U cum ssn mean performs roughly same as straight ppm
      but it does well with projected ~6.7 MAE but much better variance in preds...
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
x1 = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')
x = x1[x1.points > 0].copy()

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
# tmp = train_data.copy()
# s = CumSsnZScoreStandardizer(idcol='player_id', features=FS_POINTS)
# tmp[['player_id'] + FS_POINTS] = s.transform(tmp, handle_unseen='global').values

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

yh_train = lm.predict(y_train[FS_POINTS])
preds_proj = Model._proj(y_train[ycol].values, yh_train, preds)
dev_data['lm_preds_proj'] = preds_proj
dev_data['ytest'] = ytest

#%%
dev_data = dev_data.sort_values(by=['player_id', 'season', 'date'])
dev_data['cum_mean_points'] = (
    dev_data.groupby(['player_id', 'season'])['points']
    .expanding()
    .mean()
    .reset_index(drop=True)
    .values
)

dev_data['preds'] = (dev_data.ppm_s1 + dev_data.lm_preds_proj) * dev_data.minute_pred
#dev_data['preds'] = dev_data.points_ema_5
mae(dev_data.preds, dev_data.points)

#%%
dev_data['err'] = dev_data.points - dev_data.preds
dev_data.err.std(), dev_data.preds.var()

#%%
zz = dev_data[dev_data.player_slug==np.random.choice(dev_data.player_slug.values)].copy()

plt.clf()
plt.scatter(np.arange(zz.shape[0]),zz.points)
#plt.scatter(np.arange(zz.shape[0]),zz.preds2)
plt.scatter(np.arange(zz.shape[0]),zz.preds)
plt.plot(np.arange(zz.shape[0]),zz.preds - zz.err.std(), color='tab:red', alpha=0.2)
plt.plot(np.arange(zz.shape[0]),zz.preds + zz.err.std(), color='tab:red', alpha=0.2)
#plt.scatter(np.arange(zz.shape[0]),zz.lm_preds_proj*zz.minute_pred)
plt.plot(np.arange(zz.shape[0]),zz.cum_mean_points, ls='--', color='black')
plt.show()



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
lr = LogitModel(name='lr')
lr.build_model(**{'max_iter':100000})

#%%
tmp = train_data.copy()
s = CumSsnZScoreStandardizer(idcol='player_id', features=FS_POINTS)
s.fit(tmp[tmp.date < train_date_cutoff])
tmp[['player_id'] + FS_POINTS] = s.transform(tmp, handle_unseen='global').values

#%%
trainer = DateSplitTrainer(model=lr, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=tmp, 
        features=FS_POINTS,
        target='ppm_o_u', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date',
    )
)

#%%
err

#%%
res = []
for i in range(1000):
    x = [int(np.random.random() < 0.5) for _ in range(ytest.shape[0])]
    res.append(np.mean(x==ytest))

plt.clf()
plt.hist(res)
plt.vlines([np.mean(preds==ytest)], ymin=0, ymax=250)
plt.show()

#%%
lr.train(tmp[FS_POINTS], tmp.points_o_u)
tmp2 = test_data.copy()
tmp2[['player_id'] + FS_POINTS] = s.transform(tmp2, handle_unseen='global').values
t = lr.predict(tmp2[FS_POINTS])
tp = lr.model.predict_proba(tmp2[FS_POINTS])

#%%
np.mean(t==test_data.points_o_u.values)

#%%
ps = tp[:,0]

#%%
a = (tmp2.ppm_o_u - tmp2.ppm_o_u.mean()).values
b = ps - ps.mean()
aa = a[np.argsort(tmp2.ppm_o_u)]
bb = b[np.argsort(tmp2.ppm_o_u)]

#%%
plt.clf()
#plt.plot(aa.cumsum())
plt.plot(bb.cumsum())
plt.show()



