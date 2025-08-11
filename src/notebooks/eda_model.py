#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 19:47:29 2025

@author: rowanlavelle
"""

"""
in general, it seems unreasonable to tell when someone is going to shoot +- 2sigma 
above their current season mean (i.e, when a player is going to go off), but it seems
tractable to predict if a player is going to shoot above or below their season mean...

%% compare modeling XGB and LM by player, and by general
    - can we at least predict when a player is going to shoot above average?
            + models show no signal in predicting above or below average...
              when generally modeling, models seem to tightly follow cum ssn avg
              when modeling player by player, more variance, but worse MAE
                      
        * does standardizing data help with this?
            + predicting std does not work if using all prev hist as std...
            + if we use the minute prediction lm, to predict raw minutes
              and use an lm to predict raw points per minute and combine those
              we get down to a ~5.7 mae (on dev). then if we standardize that
              and standardize the test data (meaning mean of dev season)
              around 47% of the time we are directionaly correct on prediction
              
        * is this a classification problem better suited for logistic regression?

        
    - is it worth while to try neural nets? if XGB dsnt outperform LM maybe signal is low
        + the data seems naturally linear... you score more pts if you are shooting more
          there is not really a pocked where something would be non linear...
        
    
%% does modeling PPM and mutliplying by the minutes model have better accuracy?
    - does this provide a wider variance? how does it perform predicting above / below avg
        + slightly better MAE under ppm + minutes model... need to clean up
        
    - predict ppm O/U mean ppm * minutes?
        + directional accuracy (small signal might come here) 51% on (cum prev mean/std) std data
        
    - standardized data predicts std above mean? can use that for CI in prediction?
        
"""

#%%
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
from datetime import datetime


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

#%%
x1 = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')

#%%
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
        features=FS_MINUTE,
        target='minutes', 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err

#%%
x['minute_pred'] = minute_model.predict(x[FS_MINUTE])
y_train = x[x.date < train_date_cutoff].copy()
yh_train = minute_model.predict(y_train[FS_MINUTE])
preds_proj = Model._proj(y_train.minutes.values, yh_train, x.minute_pred)
x['minute_pred_proj'] = preds_proj

#%%
ycol = 'ppm'
test_data = x[x.date > dev_date_cutoff].copy()
dev_data = x[(x.date > train_date_cutoff) & (x.date < dev_date_cutoff)].copy()
y_train = x[x.date < train_date_cutoff].copy()
train_data = x[x.date < dev_date_cutoff].copy()

base_model = SimpleMovingAverageModel(name='ema')
base_model.build_model(window=7, avg_type='ema', source_col=ycol)

params = {
    'learning_rate': 0.05, 
    'max_depth': 5, 
    'subsample': 0.6, 
    'n_estimators': 100,
    'colsample_bytree': 0.9,
    'alpha': 10,
}

xgb = XGBModel(name='xgb')
xgb.build_model(**params)

lm = RidgeModel(name='lm')
params = {'alpha': 5}
lm.build_model(**params)

#%%
train_data = train_data.sort_values(by=['player_id', 'season', 'date'])

#%%
def train_by_player(model, trainer, fs, tag):
    groups = []
    for k,g in train_data.groupby(train_data.player_id):
        train = g[g.date < train_date_cutoff].dropna()
        test = g[g.date > train_date_cutoff].dropna()
        
        if train.shape[0] == 0 or test.shape[0] == 0:
            continue
        
        err, preds, ytest = trainer.train_and_evaluate(
            df=g,
            features=fs,
            target=ycol,
            proj=False,
            date_cutoff=train_date_cutoff,
            date_col='date'
        )
        
        
        test[f'{tag}_preds'] = preds
        yh_train = base_model.predict(train)
        preds_proj = Model._proj(train.points.values, yh_train, preds)
        test[f'{tag}_preds_proj'] = preds_proj
        
        groups.append(test)
    
    return pd.concat(groups).reset_index(drop=True)

#%%
tmp = train_by_player(base_model, trainer, ['points_ema_7'], 'bl')     
        
#%%
trainer = DateSplitTrainer(model=base_model, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=['points_ema_7'],
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date'
    )
)
print(err), preds.mean(), ytest.mean(), preds.var(), ytest.var()
dev_data['bl_preds'] = preds

yh_train = base_model.predict(y_train)
preds_proj = Model._proj(y_train.points.values, yh_train, preds)
dev_data['bl_preds_proj'] = preds_proj


#%%
trainer = DateSplitTrainer(model=xgb, metric_fn=mae)
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
dev_data['xgb_preds'] = preds
#dev_data['points_std'] = ytest

yh_train = xgb.predict(y_train[FS_POINTS])
preds_proj = Model._proj(y_train[ycol].values, yh_train, preds)
dev_data['xgb_preds_proj'] = preds_proj

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
    dev_data.groupby(['player_id', 'season'])[ycol]
    .expanding()
    .mean()
    .reset_index(drop=True)
    .values
)

#%%
dev_data['delta'] = dev_data[ycol] - dev_data.cum_mean_points
for col in ['bl_preds', 'bl_preds_proj', 'xgb_preds', 'xgb_preds_proj', 'lm_preds', 'lm_preds_proj']:
    dev_data[f'{col}_delta'] = dev_data[col] - dev_data.cum_mean_points
    print(col, np.corrcoef(dev_data.delta, dev_data[f'{col}_delta'])[0,1])
    
#%%
z = dev_data[dev_data.player_slug == 'lebron-james'].copy()
#z['color'] = z.delta.apply(lambda x: 'tab:red' if x < 0 else 'tab:green')

#%%
mae(dev_data.points, dev_data.lm_preds_proj * dev_data.minute_pred_proj)

#%%
plt.clf()
plt.plot(np.arange(z.shape[0]), z.cum_mean_points, color='black')
plt.scatter(np.arange(z.shape[0]), z[ycol], alpha=0.5)
plt.scatter(np.arange(z.shape[0]), z.xgb_preds_proj, color=z.color)
plt.show()

#%%
z['ycol_std'] = (z[ycol] - z[ycol].mean()) / z[ycol].std()

#%%
z['tmp'] = z.lm_preds * z.minute_pred
z['tmp_std'] = (z.tmp - z.tmp.mean()) / z.tmp.std()
z['pts_std'] = (z.points - z.points.mean()) / z.points.std()

#%%
plt.clf()
plt.scatter(np.arange(z.shape[0]), z.pts_std)
#plt.scatter(np.arange(z.shape[0]), z.minute_pred_proj)
plt.scatter(np.arange(z.shape[0]), z.tmp_std)
plt.show()

#%%
np.corrcoef(np.sign(z.tmp_std), np.sign(z.pts_std))

#%%
a = np.sign(z.tmp_std)
b = np.sign(z.pts_std)

#%%
(a==b).sum() / a.shape[0]

#%%
mae(dev_data[ycol], dev_data.lm_preds * dev_data.minute_pred)

#%%
plt.clf()
plt.scatter(np.arange(dev_data.shape[0]), dev_data.ppm)
#plt.scatter(np.arange(dev_data.shape[0]), dev_data.lm_preds)
plt.show()

