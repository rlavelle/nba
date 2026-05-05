#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crepated on Mon Apr 27 15:32:18 2026

TODO: 
    - rescrape all missed games
    - re run models for missed games

@author: rowanlavelle
"""

#%%
import requests

#%%
url = "https://stats.nba.com/stats/boxscoretraditionalv3?GameID=0022501147&LeagueID=00&endPeriod=0&endRange=28800&rangeType=0&startPeriod=0&startRange=0"

session = requests.Session()
session.headers.update({
  'Accept': '*/*',
  'Accept-Encoding': 'gzip, deflate, br',
  'Accept-Language': 'en-US,en;q=0.9',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive',
  'Host': 'stats.nba.com',
  'Origin': 'https://www.nba.com',
  'Pragma': 'no-cache',
  'Referer': 'https://www.nba.com/',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15'
})

#%%
resp = session.get(url)

#%%
resp.status_code

#%%
resp.json()

#%%
import pickle

#%%
x = pickle.load(open('/tmp/nc.pkl', 'rb'))
y = pickle.load(open('/tmp/df.pkl', 'rb'))
z = pickle.load(open('/tmp/df2.pkl', 'rb'))

#%%
import numpy as np
z = np.array([xx.values for xx in x.values()])

#%%
yy = pd.DataFrame(z.T, columns=list(x.keys()), index=y.index)

#%%
cols = list(x.keys())

#%%
d = pd.concat([y,yy], axis=1)

#%%
import pandas as pd
a = pd.read_pickle('data/cache/20260107_game_fts.pkl')
b = pd.read_pickle('data/cache/20260428_game_fts.pkl')

#%%
from src.utils.date import dint_to_date

b = b[b.date <= dint_to_date(20260107)].copy()
b.loc[b.date == dint_to_date(20260107), 'date'] = pd.Timestamp('2030-01-01')


#%%
a.shape, b.shape

#%%
aa = a[a.date == pd.Timestamp('2026-01-06')]
bb = b[b.date == pd.Timestamp('2026-01-06')]

#%%
aa = aa[aa.team_id.isin(bb.team_id.values)].copy()

#%%
aa = aa.sort_values(by='team_id')
bb = bb.sort_values(by='team_id')

#%%
x = aa[aa.columns[9:]]
y = bb[bb.columns[9:]]

#%%
z = x.to_numpy() - y.to_numpy()


#%%
def f(data):
    """
    Works with either a pandas DataFrame or a list of Feature objects
    """
    wanted_subs = ["_bayes_post", "_1g", "_sma_", "_ema_", "_cum_ssn_", "_hot_streak"]

    if hasattr(data, 'columns'):  # Duck typing for DataFrame
        cols = [
            col for col in data.columns
            if any(sub in col for sub in wanted_subs)
        ]
        return cols

    elif isinstance(data, list):
        cols = [
            feat.feature_name for feat in data
            if any(sub in feat.feature_name for sub in wanted_subs)
        ]
        return cols

    else:
        raise TypeError(f"Expected DataFrame or list of Feature objects, got {type(data)}")

x = aa[f(aa)]
y = bb[f(bb)]

#%%
x = x[sorted(x.columns)]
y = y[sorted(y.columns)]

#%%
z = x.to_numpy() - y.to_numpy()

#%%
import numpy as np
data = pd.DataFrame(data=np.array([np.ones(100),np.arange(0,100)]).T, columns=['id', 'x'])

import random
idxs = [random.choice(np.arange(0,100)) for _ in range(10)]
idxs

data.loc[idxs, 'x'] = np.nan

nan_filler = data.groupby(['id'])['x'].expanding().mean().shift(1)

#%%
data['res'] = (data.groupby(['id'])['x']
                .rolling(3)
                .mean()
                .shift(1)
                .fillna(nan_filler)
                .reset_index(level=0, drop=True))

data_p = data[data.index < 50].copy()

nan_filler = data_p.groupby(['id'])['x'].expanding().mean().shift(1)
nan_filler

data_p['res'] = (data_p.groupby(['id'])['x']
                .rolling(3)
                .mean()
                .shift(1)
                .fillna(nan_filler)
                .reset_index(level=0, drop=True))

#%%
tmp = data[data.index < 50].copy()

#%%
import pandas as pd
a = pd.read_pickle('data/cache/20260110_game_fts.pkl')
#b = pd.read_pickle('data/cache/20260108_game_fts.pkl')

#%%
aa = a[a.dint == 20240127]
aa.shape

#%%
bb = b[b.dint == 20251121].copy()
aa = a[a.dint == 20251121].copy()


#%%
a = a.sort_values(by=['date', 'game_id', 'team_id', 'is_home']).reset_index(drop=True)
b = b.sort_values(by=['date', 'game_id', 'team_id', 'is_home']).reset_index(drop=True)

#%%
merged = pd.DataFrame()

for col in a.columns[9:]:
    merged[col] = a[col] - b[col]
    

#%%
x = a[a.index == 262]
y = b[b.index == 262]

#%%
for col in merged.columns[0:50]:
    print(col, merged[col].sum())

#%%
import numpy as np
import pandas as pd
def div0(a, b):
    return np.where(b==0, np.nan, a/b)

a  = pd.Series(np.zeros(100))
b = pd.Series(np.ones(100))

#%%
div0(b,a)

#%%


