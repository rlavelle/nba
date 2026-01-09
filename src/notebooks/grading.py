#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 18:22:40 2025

@author: rowanlavelle
"""

#%%
import argparse
import datetime
import time

import pandas as pd

from src.db.db_manager import DBManager
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.modeling_framework.framework.dataloader import NBADataLoader
from src.types.player_types import PlayerType
from src.utils.date import date_to_dint

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib auto


#%%
date = datetime.datetime.strptime('2025-10-28', '%Y-%m-%d')
curr_date = date_to_dint(date)

dbm = DBManager()

prop_results = dbm.get_prop_results()
ml_results = dbm.get_money_line_results()

data_loader = NBADataLoader()
data_loader.load_data()

games = data_loader.get_data('games')
games = games.dropna()
games = games.sort_values(by=['team_id', 'season', 'date'])

player_data = data_loader.get_player_type(ptypes=(PlayerType.STARTER,))
player_data = player_data[~player_data.spread.isna()].copy()
player_data = player_data.drop(columns=['position'])
player_data = player_data.dropna()
player_data = player_data.sort_values(by=['player_id', 'season', 'date'])

player_data = pd.merge(prop_results, player_data,
                       on=['player_id', 'dint'],
                       how='left')

games['team_id'] = games.team_id.astype(str)
ml_results['team_id'] = ml_results.team_id.astype(str)
game_data = pd.merge(ml_results, games,
                     on=['team_id', 'dint'],
                     how='left')
    
#%%
x = games[games.team_id==15015]
x

#%%
tmp = player_data[~player_data.points.isna()].copy()
tmp['bet'] = np.where(((tmp.description == 'Over') & (tmp.preds > tmp.point)) | ((tmp.description == 'Under') & (tmp.preds < tmp.point)), 1, 0)
tmp['win'] = np.where(((tmp.description == 'Over') & (tmp.points > tmp.point)) | ((tmp.description == 'Under') & (tmp.points < tmp.point)), 1, 0)
tmp['y'] = (tmp.bet == tmp.win).astype(int)
tmp['delta'] = np.abs(tmp.preds - tmp.point)

#%%
wins = tmp[tmp.win == 1].copy()

#%%
wins.bet.mean()

#%%
results = []
max_floor = int(wins.delta.max())

for threshold in range(0, max_floor + 1):

    subset = wins[wins.delta > threshold].copy()
    subset2 = wins[(wins.delta >= threshold) & (wins.delta < threshold+1)]
    group_name = f"delta > {threshold}"
    
    mean_bet = subset.bet.mean()
    mean_bet2 = subset2.bet.mean()
    count = len(subset)
    count2 = len(subset2)
    results.append({
        'group': group_name,
        'threshold': threshold if threshold != -1 else None,
        'cum_bet': mean_bet,
        'bucket_bet': mean_bet2,
        'n1': count,
        'n2': count2
    })

# Convert to DataFrame
cumulative_groups = pd.DataFrame(results)

#%%
wins.groupby()

#%%
prev_date =datetime.date.today() - datetime.timedelta(days=1)
prev_date


#%%
a = tmp[tmp.description == 'Over'].copy()
b = tmp[tmp.description == 'Under'].copy()

a = a.sort_values(by='delta').reset_index()
a = a[a.preds > a.point].copy()
b = b.sort_values(by='delta').reset_index()
b = b[b.preds < b.point].copy()

a.y.mean(), b.y.mean()

#%%
bets = pd.concat([a,b])

#%%
bets.y.mean()

#%%
for i in range(0,int(bets.delta.max())-1):
    print(i, bets[bets.delta > i].shape[0], bets[bets.delta > i].y.mean())

#%%
a['yd'] = a.y - a.y.mean()
plt.clf()
plt.plot(a.yd.cumsum())
plt.show()

#%%
b['yd'] = b.y - b.y.mean()
plt.clf()
plt.plot(b.yd.cumsum())
plt.show()

#%%
money_line_odds = dbm.get_money_line_odds()

#%%
odds = money_line_odds[(money_line_odds.bookmaker.isin(['draftkings']))]

#%%
import datetime
date = datetime.date.today()
curr_date = date_to_dint(date)
curr_nxt_date = date_to_dint(date + datetime.timedelta(days=1))
    
#%%
curr_date, curr_nxt_date

#%%
from src.utils.date import fmt_iso_dint
odds['dint_tmp'] = odds.last_update.apply(fmt_iso_dint)
odds = odds[(odds.dint_tmp == curr_date)]

#%%
odds = odds[(odds.dint==curr_nxt_date)]

#%%
game_data = pd.merge(ml_results, games,
                     on=['team_id', 'dint'],
                     how='left')

game_data = game_data[~game_data.game_id.isna()].copy()

game_data['vegas_preds'] = 1/game_data.price

winners_idx = game_data.groupby(game_data.game_id).points.idxmax()
pred_winners_idx = game_data.groupby(game_data.game_id).preds.idxmax()
vegas_winners_idx = game_data.groupby(game_data.game_id).vegas_preds.idxmax()

# Create win column
game_data['win'] = 0
game_data.loc[winners_idx, 'win'] = 1

game_data['win_pred'] = 0
game_data.loc[pred_winners_idx, 'win_pred'] = 1


game_data['win_vegas'] = 0
game_data.loc[vegas_winners_idx, 'win_vegas'] = 1

#%%
wins = game_data[game_data.win==1].copy()
vegas_res = wins.win_vegas.mean()
model_res = wins.win_pred.mean()
n = wins.shape[0]

#%%



#%%
vegas_res, model_res, n

#%%
game_bets = game_data[game_data.win_pred==1].copy()
game_bets['y'] = game_bets.win_pred == game_bets.win

vegas_game_bets = game_data[game_data.win_vegas==1].copy()
vegas_game_bets['y'] = vegas_game_bets.win_vegas == vegas_game_bets.win

#%%
game_bets.y.mean(), vegas_game_bets.y.mean()

#%%
tmp = game_data[game_data.win_pred != game_data.win_vegas].copy()
tmp.shape

#%%
tmp[tmp.win==1].win_pred.mean(), tmp[tmp.win==1].win_vegas.mean()

#%%
stake = 1
game_bets['ret'] = np.where(game_bets.win == 1, game_bets.price * stake, 0)
game_bets['p'] = np.where(game_bets.win == 1, (game_bets.price - 1) * stake, -stake)

#%%
game_bets.ret.sum(), game_bets.p.sum(), game_bets.p.sum() / game_bets.shape[0]*stake

#%%
game_data[game_data.win==1].win_pred.mean(), game_data[game_data.win_pred==1].win.mean()

#%%
wins = game_data[game_data.win==1].copy()

#%%
wins.shape[0], game_data.shape[0]

#%%
prev_date = date_to_dint(datetime.date.today() + datetime.timedelta(days=-1))
prev_date

#%%
import os
os.pwd()

#%%
x = os.listdir(os.getcwd() + '/data/cache/')
x = [y for y in x if '.pkl' in y]
int(max(x)[:8])

#%%
a = pd.read_csv('/tmp/recent.csv')
b = pd.read_csv('/tmp/notrecent.csv')

ids = a.team_id
b = b[b.team_id.isin(ids)].copy()

#%%
z = pd.merge(a,b,on=['team_id'], how='left')

#%%
def get_ft_cols(df):
    wanted_subs = ["_bayes_post", "_1g", "_sma_", "_ema_", "_cum_ssn_", "_hot_streak"]

    cols = [
        col for col in df.columns
        if any(sub in col for sub in wanted_subs)
    ]

    return cols

fts = get_ft_cols(a)

#%%
zz = pd.DataFrame()
for col in fts:
    zz[col] = z[col + '_y'] - z[col + '_x']
    

#%%
m = zz[zz != 0]

#%%
a = pd.read_pickle('data/cache/20251028_game_fts.pkl')
b = pd.read_pickle('data/cache/20251231_game_fts.pkl')

#%%



