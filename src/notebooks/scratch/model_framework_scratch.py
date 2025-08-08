#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 18:57:44 2025

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
p.shape

#%%
m = p.groupby(['game_id', 'player_id']).minutes.mean().reset_index()

#%%
tp = m.sort_values(by='minutes', ascending=False).player_id.values[:10]

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
plt.clf()
plt.plot(t.date, t.points)
plt.show()

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
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `target` column with the next game's points for each player."""
    df = df.sort_values(['player_id', 'date'])
    df['target'] = df.groupby('player_id')['points'].shift(-1)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features like recent averages."""
    df = df.sort_values(['player_id', 'date'])

    df['points_1g'] = df.groupby('player_id')['points'].shift(1)
    df['points_3g_avg'] = df.groupby('player_id')['points'].rolling(3).mean().shift(1).reset_index(level=0, drop=True)
    df['games_played'] = df.groupby('player_id').cumcount()
    df['minutes_3g_avg'] = df.groupby('player_id')['minutes'].rolling(3).mean().shift(1).reset_index(level=0, drop=True)
    
    df['arima_minutes'] = np.nan

    for player_id, group in df.groupby('player_id'):
        minutes_series = group['minutes'].astype(float).tolist()
        arima_preds = [np.nan] * len(minutes_series)

        for i in range(5, len(minutes_series)):
            history = minutes_series[:i]
            try:
                model = ARIMA(history, order=(5, 1, 3))
                model_fit = model.fit()
                forecast = model_fit.forecast()[0]
                arima_preds[i] = forecast
            except:
                arima_preds[i] = np.nan

        df.loc[group.index, 'arima_minutes'] = arima_preds
    
    return df

def train_test_split_time(df: pd.DataFrame, date_cutoff: str):
    """Splits train/test based on date."""
    train = df[df['date'] < date_cutoff].dropna(subset=['target'])
    test = df[df['date'] >= date_cutoff].dropna(subset=['target'])
    return train, test

def train_and_evaluate(train, test, features):
    model = LinearRegression()
    model.fit(train[features], train['target'])
    
    test_preds = model.predict(test[features])
    rmse = mean_squared_error(test['target'], test_preds)
    
    return model, test_preds, rmse

#%%
from datetime import datetime
x`from sklearn.metrics import mean_squared_error

#%%
df = make_target(p)
df = make_features(df)

#%%
features = ['points_1g', 'points_3g_avg', 'games_played', 'oppDefensiveRating', 'arima_minutes']
df = df.dropna(subset=features+['target'])

train, test = train_test_split_time(df, date_cutoff=datetime(2017,10,1))

model, preds, rmse = train_and_evaluate(train, test, features)
print(f"Baseline RMSE: {rmse:.2f}")

#%%
test['preds'] = preds
t = test[test.player_id==202710]
plt.clf()
plt.plot(t.target)
plt.plot(t.preds)
plt.plot(t.points.rolling(5).mean())
plt.show()

#%%
t.loc[979:980].game_id

#%%
a = t[t.game_id=='0021700497']
b = t[t.game_id=='0021700510']

#%%
a.iloc[0]

#%%
from statsmodels.tsa.arima.model import ARIMA
# Ensure time series is sorted
t = t.sort_values(by='date').reset_index(drop=True)
t = t.replace(0, t[t.minutes > 0].minutes.mean())
series = t['minutes'].astype(float).values

history = list(series[:5])  # start with a few initial points
predictions = []

for i in range(5, len(series)):
    model = ARIMA(history, order=(5, 1, 3))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=1)
    predicted = forecast[0]
    predictions.append(predicted)

    # add the true observed value to history
    history.append(series[i])

# True future values (what we compare to)
actual = series[5:]

#%%
plt.clf()
plt.plot(actual)
plt.plot(predictions)
plt.show()

#%%
plt.clf()
plt.scatter(t.team_id, t.points)
plt.show()

#%%
np.corrcoef(t.pace, t.target)

#%%
t.columns

#%%
x = ['a', 'b', 'c']
y = ['b', 'c']
set(y) - set(x)