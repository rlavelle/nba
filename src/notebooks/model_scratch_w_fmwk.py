#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 22:36:06 2025

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

#%%

#%%
dbm = DBManager()

#%%
game_stats = dbm.get_game_stats()
games = dbm.get_games()
games = games[games.season_type_code == '00'].copy()

#%%
g = pd.merge(games, game_stats, how='left', on='game_id')

#%%
g = g[g.stat_type=='STATISTICS']

#%%
player_stats = dbm.get_player_stats()

#%%
x = g[['game_id', 'date', 'season', 'season_type_code', 'team_id', 'is_home']].copy()
p = pd.merge(x, player_stats, how='left', on=['game_id', 'team_id'])

#%%
m = p.groupby(['game_id', 'player_id']).minutes.mean().reset_index()

#%%
tp = m.sort_values(by='minutes', ascending=False).player_id.values[:10]
tp

#%%
player_meta = dbm.get_players()

#%%
top_p_minutes = player_meta[player_meta.player_id.isin(tp)].copy()

#%%
p = pd.merge(top_p_minutes,p, how='left', on='player_id')

#%%
t = p[p.player_id==202710]

#%%
t = t.sort_values(by='date')

#%%
fig, ax = plt.subplots(nrows=1,ncols=5)
axs = ax.flatten()
for i,s in enumerate(t.season.unique()):
    tmp = t[t.season==s]
    tmp = tmp.sort_values(by='date')
    axs[i].plot(tmp.date, tmp.points)
    axs[i].set_title(s)

plt.show()

#%%
t.groupby(t.season).game_id.nunique()

#%%
plt.clf()
plt.plot(t.points)
plt.plot(t.points.rolling(window=5).mean())
plt.show()

#%%
x = g[['game_id', 'date', 'season', 'season_type_code', 'team_id', 'is_home', 'defensiveRating']].copy()
x = x.rename(columns={'defensiveRating': 'oppDefensiveRating'})
pp = p.copy()
a = p.shape[0]
p = pd.merge(p,x, how='left', on=['game_id', 'team_id', 'season', 'season_type_code', 'date'])
b = p.shape[0]
assert a == b, 'fuck'

#%%
import numpy as np
from typing import Union
from src.modeling_framework.framework.model import Model

class LastGameModel(Model):
  
    def build_model(self, **params):
        pass
        
    def fit(self, X_train, y_train):
        pass
    
    def predict(self, X) -> Union[float, np.ndarray]:
        # X should contain the last game's stats
        if isinstance(X, pd.DataFrame):
            return X['points_1g'].values
        return X  # Assuming X is already the last game points

class RollingAverageModel(Model):
    
    def build_model(self, **params):
        self.window = params['window']
    
    def fit(self, X_train, y_train):
        pass
    
    def predict(self, X) -> Union[float, np.ndarray]:
        # X should contain the rolling average column
        col_name = f'points_{self.window}g_avg'
        if isinstance(X, pd.DataFrame):
            return X[col_name].values
        return X  # Fallback if passing pre-computed values

    
#%%
from src.feature_engineering.last_game_value import LastGameValueFeature
from src.feature_engineering.moving_avg import SimpleMovingAvgFeature, ExponentialMovingAvgFeature
from src.feature_engineering.base import FeaturePipeline

pp = p.copy()

# Create features first
features = [
    LastGameValueFeature(),
    SimpleMovingAvgFeature(window=3),
    SimpleMovingAvgFeature(window=5),
    SimpleMovingAvgFeature(window=10),
    ExponentialMovingAvgFeature(span=7)
]

pipeline = FeaturePipeline(features)
df = pipeline.transform(p)

# Initialize models
last_game_model = LastGameModel(name = "last_game_model")

three_avg_model = RollingAverageModel(name = "rolling_3avg_model") 
three_avg_model.build_model(window=3)

five_avg_model = RollingAverageModel(name = "rolling_5avg_model")
five_avg_model.build_model(window=5)

ten_avg_model = RollingAverageModel(name = "rolling_10vg_model")
ten_avg_model.build_model(window=10)


# Example evaluation
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Prepare test data (filter most recent games)
test_data = df.dropna(subset=['points_1g', 'points_3g_avg', 'points_5g_avg', 'points_10g_avg', 'points'])

# Evaluate
models = [last_game_model, three_avg_model, five_avg_model, ten_avg_model]
for model in models:
    score = model.evaluate(
        X_val=test_data,
        y_val=test_data['points'],
        metric_fn=rmse
    )
    print(f"{model.name} RMSE: {score:.2f}")
    

#%%
model.stats(test_data, test_data['points'])

#%%
X_preds = ten_avg_model.predict(test_data)
X_preds

yh = X_preds
yt = test_data.points.values

print(yh.mean(), yt.mean(), yh.var(), yt.var())

yyh = yh - yh.mean()
yyt = yt - yt.mean()

b = (yyh.T@yyt) / (yyh.T@yyh)


xx = (b*yyh)[np.argsort(yt)]
yy = yyt[np.argsort(yt)]

plt.clf()
plt.plot(np.cumsum(xx))
plt.plot(np.cumsum(yy))
plt.show()

#%%
yc = yt.mean() + b*(yh - yh.mean())
yf = yt.mean() + (yt.std()/yc.std()) * (yc - yt.mean())

#%%
print(yf.mean(), yt.mean(), yf.var(), yt.var())

#%%
xx = (yc-yc.mean())[np.argsort(yt)]
zz = (yf-yf.mean())[np.argsort(yt)]
yy = (yt-yt.mean())[np.argsort(yt)]
plt.figure(1)
plt.clf()
plt.plot(np.cumsum(xx))
plt.plot(np.cumsum(zz))
plt.plot(np.cumsum(yy))
plt.show()

#%%


#%%
np.corrcoef(np.sign(np.diff(x)), np.sign(np.diff(y)))

#%%
from src.modeling_framework.models.xgb_model import XGBModel
from src.modeling_framework.models.regression import LinearModel

model = LinearModel(name='lm')
params = {'learning_rate': 0.01, 'max_depth':10, 'subsample': 0.6, 'n_estimators': 100}
model.build_model()

#%%
from src.modeling_framework.trainers.date_split_trainer import DateSplitTrainer
from datetime import datetime
date_cutoff = datetime(2024, 10, 1)
trainer = DateSplitTrainer(model=model, metric_fn=rmse)
err, preds, ytest = trainer.train_and_evaluate(df=df[(df.date > datetime(2021,10,1)) & (df.points > 0)], 
                                                 features=pipeline.get_feature_names(), 
                                                 target='points', 
                                                 proj=False, 
                                                 date_cutoff=date_cutoff,
                                                 date_col='date')
err, preds.mean(), ytest.mean(), preds.var(), ytest.var()

#%%
yh = preds - preds.mean()
yt = ytest - ytest.mean()
yh.shape, yt.shape

#%%
xx = yh[np.argsort(ytest)]
yy = yt[np.argsort(ytest)]
plt.clf()
plt.plot(np.cumsum(xx))
plt.plot(np.cumsum(yy))
plt.show()

#%%
ytest.max(), preds.max()

#%%
from src.modeling_framework.framework.gridsearch import GridSearch
gs = GridSearch(model=model, 
                trainer=trainer, 
                param_grid={
                    'learning_rate': [00.001, .01, 0.05, 0.1],
                    'max_depth': [5,10,25,50],
                    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    'n_estimators': list(range(10,250, 25)),
                    'lambda': list(range(1,10,1))
                })

m,prm,s = gs.search(df=df, features=pipeline.get_feature_names(), target='points', proj=False, date_cutoff=date_cutoff, date_col='date')

#%%


