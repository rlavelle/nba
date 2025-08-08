#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 23:08:12 2025

@author: rowanlavelle

weird corr found with bug in bayesian code (not cheating, just does not make sense)
 - if you use the previous games {f} 
     and weight it at the current game in the season,
     assume strong prior built on the previos season you get strong signal...
"""

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
data_loader = NBADataLoader()
data_loader.load_data()

#%%
tmp = data_loader.get_player_type(ptypes=[PlayerType.STARTER])

#%%
x = tmp.copy()
x = x[~x.spread.isna()].copy()
x = x.drop(columns=['position'])
x = x.dropna()
x = x.sort_values(by=['player_id', 'season', 'date'])

for f in PLAYER_FEATURES:
    print(f)
    features = [
        ExponentialMovingAvgFeature(span=7, source_col=f),  
        ExponentialMovingAvgFeature(span=5, source_col=f),
        ExponentialMovingAvgFeature(span=3, source_col=f),
        CumSeasonAvgFeature(source_col=f),
        CumSeasonEMAFeature(source_col=f),
        SimpleMovingAvgFeature(window=3, source_col=f),
        SimpleMovingAvgFeature(window=5, source_col=f),
        LastGameValueFeature(source_col=f)
    ]
    
    dependents = []
    for feature in features:
        if '_cum_ssn' in feature.feature_name:
            # can this be an optimization problem? best combo of bayes / hot with simple features
            dependents.append(
                BayesPosteriorFeature(ybar_col=feature.feature_name,
                                      source_col=f)
            )
            
        if '3g' not in feature.feature_name: 
            dependents.append(
                PlayerHotStreakFeature(comp_col=feature.feature_name,
                                       source_col=f)
            )
        
    features.extend(dependents)
    pipeline = FeaturePipeline(features)
    x = pipeline.transform(x)

x = x.dropna()
x = x[x['season'] > x['season'].min()].copy()
x.season.unique()

#%%
x.to_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/feature_df.pckl')

#%%
ycol = 'minutes'
results = []

for target_feature in PLAYER_FEATURES:
    feature_cols = [col for col in x.columns 
                   if col.startswith(target_feature) and col != target_feature and '_' in col]
    
    for feature_col in feature_cols:
        def calc_corr(group):
            return np.corrcoef(group[feature_col], group[ycol])[0, 1]
        
        mean_corr = x.groupby('player_id').apply(calc_corr, include_groups=False).mean()
        
        results.append({
            'source_column': target_feature,
            'feature_type': '_'.join(feature_col.split('_')[-2:]),
            'feature_name': feature_col,
            'mean_correlation': mean_corr
        })

# Convert to DataFrame and sort by correlation strength
correlation_results = (
    pd.DataFrame(results)
    .sort_values('mean_correlation', key=abs, ascending=False)
    .reset_index(drop=True)
)

#%%
t = correlation_results[correlation_results.mean_correlation > 0.1].copy()
t = t.sort_values(by=['source_column', 'mean_correlation'], ascending=False)

#%%
minutes_best_cols = t.feature_name.unique()
# points_best_cols = t.feature_name.unique()

#%%
z = x[x.player_slug == 'james-harden'].copy()
z = z[z.season=='2024-25']
z = z[['player_id', 'game_id', 'season', 'date', 'points', 'points_sma_3']]

#%%
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel

model = LinearModel(name='lm')
model.build_model()

#%%
from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from datetime import datetime
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)
test_data = x[x.date > dev_date_cutoff].copy()
train_data = x[x.date < dev_date_cutoff].copy()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

base_model = SimpleMovingAverageModel(name='ema')
base_model.build_model(window=7, avg_type='ema', source_col=ycol)

# base model
trainer = DateSplitTrainer(model=base_model, metric_fn=mae)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=[c for c in train_data.columns 
                  if '_ema_' in c 
                  and 'bayes' not in c 
                  and 'hot_streak' not in c], 
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()


#%%
# base ema only model all cols
trainer = DateSplitTrainer(model=model, metric_fn=rmse)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=[f'{c}_ema_{i}' for i in [3,5,7] for c in minutes_best_cols], 
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
# ema and hot streak all cols
trainer = DateSplitTrainer(model=model, metric_fn=rmse)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=train_data, 
        features=[f'{c}_ema_{i}' for i in [3,5,7] for c in minutes_best_cols] + 
                 [f'{c}_3g_v_{c}_ema_{i}_hot_streak' for i in [5,7] for c in minutes_best_cols], 
        target=ycol, 
        proj=False, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
def build_quick_lm(fts):
    trainer = DateSplitTrainer(model=model, metric_fn=rmse)
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
    #err, preds.mean(), ytest.mean(), preds.var(), ytest.var()
    return err

#%%
#cols = list(points_best_cols)
cols = list(minutes_best_cols)

curr_order = [([], np.inf), ]

#%%
while len(cols) > 0:
    best_col = None
    best_err = np.inf
    for col in cols:
        e = build_quick_lm(curr_order[-1][0] + [col])
        if e < best_err:
            best_col = col
            best_err = e
            
    curr_order.append((
        curr_order[-1][0] + [best_col], best_err
    ))
    cols.remove(best_col)
    print(len(cols), curr_order[-1][1])

#%%
errs = [e[1] for e in curr_order][1:]
plt.clf()
plt.plot(errs)
plt.show()

#%%
#curr_order[30][0]
# curr_order[27][0]
curr_order[-1][0]

#%%
errs[-1]

#%%