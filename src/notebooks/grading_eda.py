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
player_data['delta'] = player_data.preds - player_data.point

#%%
x = player_data.groupby([player_data.player_id, player_data.player_slug]).delta.mean()

#%%
player_data.player_slug.unique()