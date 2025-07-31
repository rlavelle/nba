#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:50:34 2025

@author: rowanlavelle

Bayesian play:
    
    prior mu0 and tau0 where mu ~ N(mu0, tau0)
    observered xbar and sbar
    
    posterior
    mun = (n * xbar/sbar + mu0/tau0) / (n/sbar + 1/tau0)
    taun = 1/(n/sbar + 1/tau0)
    
    predictive x ~ N(mun, sbar+taun)
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
game_meta = dbm.get_games()
game_meta = game_meta[game_meta.season_type_code == '00'].copy()
games = pd.merge(game_meta, game_stats, how='left', on='game_id')
games = games[games.stat_type=='STATISTICS']

#%%
player_stats = dbm.get_player_stats()
player_meta = dbm.get_players()
players = pd.merge(player_stats, player_meta, how='left', on='player_id')

x = games[['game_id', 'team_id', 'season', 'season_type', 'season_type_code',
            'dint', 'date', 'is_home']]

players = pd.merge(x, players, on=['game_id', 'team_id'], how='left')

#%%
tmp = pd.merge(games[games.is_home==1][['game_id', 'points']],
               games[games.is_home==0][['game_id', 'points']],
               on='game_id', how='left')
tmp['spread'] = np.abs(tmp.points_x - tmp.points_y)

players = pd.merge(players, tmp[['game_id', 'spread']], on='game_id', how='left')

mins = players.groupby(['team_id', 'season', 'player_id']).minutes.agg(mu_m='mean', var_m='var').reset_index(drop=False)
mins = mins.sort_values(by=['team_id', 'season', 'mu_m'])

mins['rk'] = mins.groupby([mins.team_id, mins.season]).mu_m.rank(method='first', ascending=False)
mins['player_type'] = np.where(mins.mu_m >= 28, 'S1', 
                               np.where((mins.mu_m >= 20) & (mins.mu_m < 28), 'PB', 
                                        np.where((mins.mu_m >= 10) & (mins.mu_m < 20), 'SB', 'B')))
mins.loc[mins.rk <= 5, 'player_type'] = 'S'
players = pd.merge(players, mins, on=['player_id', 'team_id', 'season'], how='left')

#%%
x = players[players.player_type.isin(['S'])].copy()
x['ppm'] = np.where(x.minutes==0, 0, x.points/x.minutes)
x = x[~x.spread.isna()].copy()

#%%
x = x.sort_values(by=['player_id', 'season', 'date'])

#%%
p = 'james-harden'#np.random.choice(x.player_id.values)
p = x[x.player_slug == p]

#%%
plt.clf()
for k,g in p.groupby(p.season):
    plt.hist(g.points, bins=30, alpha=0.5)
plt.show()

#%%
plt.clf()
plt.hist(p[p.season=='2015-16'].points, bins=30, alpha=0.5)
plt.show()

#%%
s1 = p[p.season=='2017-18'].copy()
s2 = p[p.season=='2018-19'].copy()

#%%
norm = np.random.normal

#%%
mu0 = s1.points.mean()
tau20 = s1.points.var() / s1.shape[0]
var = s1.points.var()#p[p.season <= '2017-18'].points.var()

#%%
mu_c = mu0
s2_c = tau20

print(f'mu = {round(mu_c, 2)} var = {round(s2_c, 2)}')

for i in range(0, s2.shape[0]):
    x = s2.iloc[i].points
    mu_n = ( (1/var)*x + (1/s2_c)*mu_c ) / ( (1/var) + (1/s2_c) )
    s2_n = 1 / ((1/var) + (1/s2_c))
    
    mu_c = mu_n
    s2_c = s2_n

    print(f'n = {i} | points = {x} mu = {round(mu_c, 2)} var = {round(s2_c, 2)}')

#%%
xt = norm(mu_c, np.sqrt(s2_c + var), size=(s2.shape[0]))
plt.clf()
plt.hist(s2.points, bins=20, alpha=0.5)
plt.hist(xt, bins=20, alpha=0.5)
plt.show()

#%%

np.clip