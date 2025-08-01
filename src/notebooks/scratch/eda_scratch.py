#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:51:23 2025

@author: rowanlavelle

Questions:

    %% how to label starters - S1 - bench - reserve (labels are season dependent, given modeling will box seasons)
        -- S, S1, PB, SB, B (starters, sem starters, primary bench, secondary bench, deep bench)
    
    %% does spread effect starters and S1 minutes? ppm?
        -- players play less minutes as spread increases (makes sense give PB,SB,B more play time) but no corr on PPM
        -- if the spread is high (> 20 or so) and the starting player is player high minutes (> 25-28) PPM starts to drop
        -- spread has no direct corr at large to points (-0.09) or ppm
        
    %% what extended box score features correlate to ppm? to score?
        -- hard to corr to ppm, but raw score has corr for several features (> 0.1)
        
    %% what opposing team box score features correlate to ppm? to score? (does this need to be season avg?)
        -- opp team box score is not predictive (defensiveRating highest, but < 0.5)
    
    is it possible to match players based on position - then use match up box score features to correlate ppm / score


Notes (FOR STARTERS):
    %% there is some corr between the raw box score data (cum avg of season as well as 3 game streak above/below avg). Minutes is high corr
    %% ppm when a player is heating up has a slight corr to points
    %% nothing seems to corr to ppm, and its not a very descriptive metric, looks like modeling raw points is the way to go
    %% minutes has similar predictors to points, which makes sense. The hot streaks has much higher corr than cum ssn avg, this makes sense
    %% interesting that attempted fg is a predictor, but makes sense that the more a player shoots the more points they'll score
    %% doesnt matter how own team is performing in long and short term regarding corr to points
    %% the opposing team doesnt seem to have a strong corr to points either
    %% does this really matter no one playes defense in the regular season
    
    %% bayes shows a bit more corr than cum mean, but does not imporve streak feature
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
corr = lambda x: np.corrcoef(x.spread, x.minutes)[0,1]
x.groupby(x.player_id).apply(corr, include_groups=False).mean()

#%%
plt.clf()
plt.scatter(x.spread, x.ppm)
plt.show()

#%%
plt.clf()
b = (x.spread >= 20) & (x.minutes >= 30)
plt.scatter(x[~b].minutes, x[~b].ppm, alpha=0.5)
plt.scatter(x[b].minutes, x[b].ppm, alpha=0.5)
plt.show()

#%%
x[~b]

#%%
features = ['minutes', 'fieldGoalsMade', 'ppm',
'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade',
'threePointersAttempted', 'threePointersPercentage', 'freeThrowsMade',
'freeThrowsAttempted', 'freeThrowsPercentage', 'reboundsOffensive',
'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks',
'blocksAgainst', 'turnovers', 'foulsPersonal', 'foulsDrawn', 'points',
'plusMinusPoints', 'offensiveRating', 'estimatedOffensiveRating',
'defensiveRating', 'estimatedDefensiveRating', 'netRating',
'estimatedNetRating', 'assistPercentage', 'assistToTurnover',
'assistRatio', 'turnoverRatio', 'offensiveReboundPercentage',
'defensiveReboundPercentage', 'reboundPercentage',
'effectiveFieldGoalPercentage', 'trueShootingPercentage',
'usagePercentage', 'estimatedUsagePercentage', 'pace', 'estimatedPace',
'pacePer40', 'possessions', 'PIE', 'percentageFieldGoalsAttempted2pt',
'percentageFieldGoalsAttempted3pt', 'percentagePoints2pt',
'percentagePointsMidrange2pt', 'percentagePoints3pt',
'percentagePointsFastBreak', 'percentagePointsFreeThrow',
'percentagePointsOffTurnovers', 'percentagePointsPaint',
'freeThrowAttemptRate', 'percentageAssisted2pt',
'percentageUnassisted2pt', 'percentageAssisted3pt',
'percentageUnassisted3pt', 'percentageAssistedFGM',
'percentageUnassistedFGM', 'pointsOffTurnovers', 'pointsSecondChance',
'pointsFastBreak', 'pointsPaint', 'oppPointsOffTurnovers',
'oppPointsSecondChance', 'oppPointsFastBreak', 'oppPointsPaint',
'oppEffectiveFieldGoalPercentage', 'oppFreeThrowAttemptRate',
'oppTeamTurnoverPercentage', 'oppOffensiveReboundPercentage',
'teamTurnoverPercentage', 'percentagePersonalFouls',
'percentagePersonalFoulsDrawn']

#%%
x = players[players.player_type.isin(['S'])].copy()
x['ppm'] = np.where(x.minutes==0, 0, x.points/x.minutes)
x = x[~x.spread.isna()].copy()
x = x.drop(columns=['position'])
x = x.dropna()

x = x.sort_values(by=['player_id', 'season', 'date'])

for f in features:
    x['cum_f'] = x.groupby([x.player_id, x.season])[f].cumsum()
    x['n'] = x.groupby([x.player_id, x.season])[f].cumcount() + 1
    x[f'{f}_f'] = (x.cum_f / x.n).shift(1) # lol
    
    x['avg'] = x.groupby([x.player_id, x.season])[f].rolling(3).mean().shift(1).values
    x[f'{f}_hot'] = np.where(x[f'{f}_f']==0, 1, x.avg / x[f'{f}_f'])
    x[f'{f}_hot'] = x[f'{f}_hot'].fillna(1) 
    
    season_stats = x.groupby([x.player_id, x.season])[f].agg(['mean', 'var', 'size']).reset_index()
    season_stats = season_stats.rename(columns={'mean': f'{f}_mean', 'var': f'{f}_var', 'size':f'{f}_N'})
    season_stats[f'{f}_mu0'] = season_stats.groupby('player_id')[f'{f}_mean'].shift(1)
    season_stats[f'{f}_s2bar'] = season_stats.groupby('player_id')[f'{f}_var'].shift(1)

    season_stats[f'{f}_tau20'] = (
        season_stats.groupby('player_id')[f'{f}_mean']
        .expanding()  # Get all previous values for each row
        .var()  # Calculate sample variance
        .reset_index(level=0, drop=True)
    )*1000000
    
    x = x.merge(season_stats[['player_id', 'season', f'{f}_mu0', f'{f}_s2bar', f'{f}_tau20', f'{f}_N']], 
                 on=['player_id', 'season'], how='left')
    
    
    x[f'{f}_mu_n'] = (
        (
            (x.n.shift(1)*x[f'{f}_f'])/x[f'{f}_s2bar'] + 
            x[f'{f}_mu0']/x[f'{f}_tau20']
        ) / 
        (
            x.n.shift(1)/x[f'{f}_s2bar'] + 1/x[f'{f}_tau20']
        )
    )
    
    x[f'{f}_var_n'] = 1 / (x.n.shift(1)/x[f'{f}_s2bar'] + 1/x[f'{f}_tau20'])
    x[f'{f}_n'] = x.n
    
    x[f'{f}_hot_bayes'] = np.where(x[f'{f}_mu_n']==0, 1, x.avg / x[f'{f}_mu_n'])
    x[f'{f}_hot_bayes'] = x[f'{f}_hot_bayes'].fillna(1) 

    
x = x.drop(columns=['cum_f', 'n', 'avg']).dropna()
x = x[x['season'] > x['season'].min()].copy()

a = []
for f in features:
    corr1 = lambda z: float(np.corrcoef(z[f'{f}_f'], z.points)[0,1]) 
    corr2 = lambda z: float(np.corrcoef(z[f'{f}_hot'], z.points)[0,1])
    corr3 = lambda z: float(np.corrcoef(z[f'{f}_mu_n'], z.points)[0,1])
    corr4 = lambda z: float(np.corrcoef(z[f'{f}_hot_bayes'], z.points)[0,1])

    t1 = x.groupby(x.player_id).apply(corr1, include_groups=False).mean()
    t2 = x.groupby(x.player_id).apply(corr2, include_groups=False).mean()
    t3 = x.groupby(x.player_id).apply(corr3, include_groups=False).mean()
    t4 = x.groupby(x.player_id).apply(corr4, include_groups=False).mean()
            
    a.append((f, t1, t2, t3, t4))

a = sorted(a, key=lambda t: t[1])
a = pd.DataFrame(a, columns=['col', 'rho_f', 'rho_hot', 'bayes_mu', 'bayes_hot'])

#%%
plt.clf()
plt.scatter(x.possessions_hot, x.points)
plt.show()

#%%
x.PIE_f

#%%
import random
pids = np.random.choice(x.player_id, size=5)

plt.clf()
for p in pids:
    plt.hist(x[x.player_id==p].foulsDrawn, bins=50, alpha=0.5)
plt.show()

#%%
from scipy.stats import t

for col in {'rho_f', 'rho_hot', 'bayes_mu', 'bayes_hot'}:
    tmp = a[col]*np.sqrt(x.shape[0] - 2) / np.sqrt(1-a[col]**2)
    p_value = 2 * t.sf(np.abs(tmp), x.shape[0]-2)
    a[f'{col}_p'] = p_value
    
#%%
t = x.groupby(x.player_id).apply(corr3, include_groups=False).reset_index(drop=False)
t = t.sort_values(by=0, ascending=False)
pp = t.player_id.values[:9]

#%%
# 1641706 - stron corr 

#%%
# sum((xd*yd))/(root(xd**2)*root(yd**2))

#%%
fig, ax = plt.subplots(3,3)
axs = ax.flatten()
i = 0
f = 'minutes_mu_n'

for k,g in x[x.player_id.isin(pp)].groupby(x.player_id):
    xd = g[f] - g[f].mean()
    yd = g.points - g.points.mean()
    c = np.std(g[f])*np.std(g.points) * g.shape[0]
    num = np.cumsum(xd*yd)
    print(num.values[-1]/c)
    print(k)
    #axs[i].scatter(g[f], g.points)
    axs[i].plot(num/c)
    axs[i].set_title(k)
    i += 1
    
fig.tight_layout()
plt.show()
    
