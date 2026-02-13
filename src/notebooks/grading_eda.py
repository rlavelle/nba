#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 11:25:19 2026

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
dbm = DBManager()

prop_results = dbm.get_prop_results()
ml_results = dbm.get_money_line_results()

data_loader = NBADataLoader()
data_loader.load_data()

games = data_loader.get_data('games')
games = games.sort_values(by=['team_id', 'season', 'date'])

player_data = data_loader.get_player_type(ptypes=(PlayerType.STARTER, PlayerType.STARTER_PLUS, PlayerType.PRIMARY_BENCH))
player_data = player_data[~player_data.spread.isna()].copy()
player_data = player_data.drop(columns=['position'])
player_data = player_data.sort_values(by=['player_id', 'season', 'date'])

player_results = pd.merge(prop_results, player_data,
                       on=['player_id', 'dint'],
                       how='left')

games['team_id'] = games.team_id.astype(str)
ml_results['team_id'] = ml_results.team_id.astype(str)
game_data = pd.merge(ml_results, games,
                     on=['team_id', 'dint'],
                     how='left')

#%%
player_data.player_type.unique()

#%%
player_results['delta'] = player_results.preds - player_results.point

#%%
x = player_results.groupby([player_results.player_id, player_results.player_slug]).delta.mean()

#%%
z = player_results[player_results.player_slug == 'kelly-oubre-jr'].copy()

#%%
player_data[player_data.season == player_data.season.max()].player_type.values[0]

#%%
x = player_results[player_results.player_id == 203954]
xx = player_data[player_data.player_id == 203954]

#%%
a = games[games.dint == 20251226]

#%%%
203954
203935

#%%
a = pd.read_pickle('data/cache/20260107_player_fts.pkl')

#%%
a.player_type.unique()

#%%
game_data = game_data[~game_data.game_id.isna()].copy()
game_data['vegas_preds'] = 1.0 / game_data.price

winners_idx = game_data.groupby(game_data.game_id).points.idxmax()
pred_winners_idx = game_data.groupby(game_data.game_id).preds.idxmax()
vegas_winners_idx = game_data.groupby(game_data.game_id).vegas_preds.idxmax()

game_data['win'] = 0
game_data.loc[winners_idx, 'win'] = 1

game_data['win_pred'] = 0
game_data.loc[pred_winners_idx, 'win_pred'] = 1

game_data['win_vegas'] = 0
game_data.loc[vegas_winners_idx, 'win_vegas'] = 1
    
game_wins = game_data[game_data.win == 1].copy()

#%%
game_wins.win.mean()


