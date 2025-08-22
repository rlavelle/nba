#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:36:01 2025

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
from src.modeling_framework.standardizers.zscore import ZScoreStandardizer

#%%
data_loader = NBADataLoader()
data_loader.load_data()

#%%
games = data_loader.get_data('games')

#%%
x = games.copy()
x = x.dropna()
x = x.sort_values(by=['team_id', 'season', 'date'])

for f in GAME_FEATURES:
    print(f)
    features = [
        ExponentialMovingAvgFeature(span=7, source_col=f, group_col=('team_id',)),  
        ExponentialMovingAvgFeature(span=5, source_col=f, group_col=('team_id',)),
        ExponentialMovingAvgFeature(span=3, source_col=f, group_col=('team_id',)),
        CumSeasonAvgFeature(source_col=f, group_col=('team_id','season')),
        CumSeasonEMAFeature(source_col=f, group_col=('team_id','season')),
        SimpleMovingAvgFeature(window=3, source_col=f, group_col=('team_id',)),
        SimpleMovingAvgFeature(window=5, source_col=f, group_col=('team_id',)),
        LastGameValueFeature(source_col=f, group_col=('team_id',))
    ]
    
    dependents = []
    for feature in features:
        if '_cum_ssn' in feature.feature_name:
            # can this be an optimization problem? best combo of bayes / hot with simple features
            dependents.append(
                BayesPosteriorFeature(ybar_col=feature.feature_name,
                                      source_col=f,
                                      id_col='team_id',
                                      group_col=('team_id','season'))
            )
            
        if '3g' not in feature.feature_name and '1g' not in feature.feature_name: 
            dependents.append(
                PlayerHotStreakFeature(window=1,
                                       comp_col=feature.feature_name,
                                       source_col=f,
                                       group_col=('team_id','season'))
            )
            
            dependents.append(
                PlayerHotStreakFeature(window=3,
                                       comp_col=feature.feature_name,
                                       source_col=f,
                                       group_col=('team_id','season'))
            )
        
        if '3g' in feature.feature_name:
             dependents.append(
                 PlayerHotStreakFeature(window=1,
                                        comp_col=feature.feature_name,
                                        source_col=f,
                                        group_col=('team_id','season'))
             )
                    
    features.extend(dependents)
    pipeline = FeaturePipeline(features)
    x = pipeline.transform(x, sort_order=('team_id', 'season', 'date'))

x = x.dropna()
x = x[x['season'] > x['season'].min()].copy()
x.season.unique()

#%%
x.to_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/game_feature_df.pckl')

#%%
x = pd.read_pickle('/Users/rowanlavelle/Documents/Projects/nba/data/tmp/game_feature_df.pckl')

#%%
meta_cols = [
    "season", "season_type", "season_type_code", "dint",
    "date", "team_id", "is_home", "stat_type"
]

home = x[x["is_home"] == 1].set_index("game_id")
away = x[x["is_home"] == 0].set_index("game_id")
meta = home[meta_cols].reset_index()

home_stats = home.drop(columns=meta_cols)
away_stats = away.drop(columns=meta_cols)

diff_stats = (home_stats - away_stats).abs()
diff_stats = diff_stats.reset_index()

df_diff = pd.concat([meta, diff_stats.drop(columns=["game_id"])], axis=1)

#%%
ycol = 'points'
results = []

for target_feature in GAME_FEATURES:
    feature_cols = [col for col in x.columns 
                   if col.startswith(target_feature) and col != target_feature and '_' in col]
    
    for feature_col in feature_cols:
        def calc_corr(group):
            return np.corrcoef(group[feature_col], group[ycol])[0, 1]
        
        mean_corr = df_diff.groupby('team_id').apply(calc_corr, include_groups=False).mean()
        
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
fts = correlation_results.feature_name.unique()

#%%
df_diff['spread'] = df_diff.points.abs()

#%%
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel
from src.modeling_framework.models.base_models import SimpleMovingAverageModel

from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from datetime import datetime
train_date_cutoff = datetime(2023, 10, 1)
dev_date_cutoff = datetime(2024, 10, 1)
test_data = df_diff[df_diff.date > dev_date_cutoff].copy()
dev_data = df_diff[(df_diff.date > train_date_cutoff) & (df_diff.date < dev_date_cutoff)].copy()
train_data = df_diff[df_diff.date < dev_date_cutoff].copy()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

model = LinearModel(name='lm')
model.build_model()

wanted_subs = ["_bayes_post", "_1g", "_sma_", "_ema_", "_cum_ssn_", "_hot_streak"]

cols = [
    col for col in df_diff.columns
    if any(sub in col for sub in wanted_subs)
]

#%%
def build_quick_lm(fts):
    trainer = DateSplitTrainer(model=model, metric_fn=rmse)
    err, preds, ytest = (
        trainer
        .train_and_evaluate(
            df=train_data, 
            features=fts, 
            target='spread', 
            proj=False, 
            date_cutoff=train_date_cutoff,
            date_col='date')
    )
    return err

def bic(model, X, y):
    yh = model.predict(X)
    rss = np.sum((y-yh)**2)
    n = len(y)
    k = X.shape[1]+1
    return n * np.log(rss/n) + k * np.log(n)

#%%
print(len(cols))
curr_order = [([], np.inf), ]

#%%
prev_bic = 0
while len(cols) > 0:
    best_col = None
    best_err = np.inf
    for col in cols:
        fts = curr_order[-1][0] + [col]
        e = build_quick_lm(fts)
        if e < best_err:
            best_col = col
            best_err = e
            bic_v = bic(model, train_data[fts], train_data['spread'])
    
    print(bic_v - prev_bic)
    prev_bic = bic_v
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
params = {
    'learning_rate': 0.0, 
    'max_depth': 10, 
    'subsample': 0.8, 
    'n_estimators': 300,
    'colsample_bytree': 0.5,
    'alpha': 0.05,
    'lambda': 0.05,
}

xgb = XGBModel(name='xgb')
xgb.build_model(**params)

#%%
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

#%%
tmp = train_data.copy()
s = ZScoreStandardizer(idcol=None, features=['spread'])
s.fit(tmp[tmp.date < train_date_cutoff])
tmp['spread'] = s.transform(tmp, handle_unseen='global').values

#%%
trainer = DateSplitTrainer(model=xgb, metric_fn=rmse)
err, preds, ytest = (
    trainer
    .train_and_evaluate(
        df=tmp, 
        features=cols, 
        target='spread', 
        proj=True, 
        date_cutoff=train_date_cutoff,
        date_col='date')
)

#%%
a = ytest - ytest.mean()
b = preds - preds.mean()
aa = a[np.argsort(ytest)]
bb = b[np.argsort(ytest)]
plt.clf()
plt.plot(aa.cumsum())
plt.plot(bb.cumsum())
plt.show()

#%%
plt.clf()
plt.scatter(np.arange(ytest.shape[0]), ytest)
plt.scatter(np.arange(preds.shape[0]), preds)
plt.show()

#%%
dev_data['ytest'] = (ytest*train_data.spread.std()) + train_data.spread.mean()
dev_data['preds'] = (preds*train_data.spread.std()) + train_data.spread.mean()

#%%
a = (dev_data.ytest - dev_data.ytest.mean()).values
b = (dev_data.preds - dev_data.preds.mean()).values
aa = a[np.argsort(dev_data.ytest)]
bb = b[np.argsort(dev_data.ytest)]
plt.clf()
plt.plot(aa.cumsum())
plt.plot(bb.cumsum())
plt.show()

#%%
plt.clf()
plt.scatter(np.arange(ytest.shape[0]), dev_data.ytest)
plt.scatter(np.arange(preds.shape[0]), dev_data.preds)
plt.show()

#%%
mae(dev_data.ytest, dev_data.preds)

#%%
dev_data.preds.var()

#%%
dev_data.ytest.median()

#%%
plt.clf()
plt.hist(dev_data.spread, bins=50,alpha=0.5)
plt.hist(dev_data.preds, bins=50,alpha=0.5)
plt.show()



