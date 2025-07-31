# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
ROOT = '/Users/rowanlavelle/nba/'
BASDIR = ROOT+'data/'

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import configparser
sns.set_theme()

#%%


#%%
# "https://stats.nba.com/stats/boxscoretraditionalv3"
# "https://stats.nba.com/stats/boxscoreadvancedv3?"
# "https://stats.nba.com/stats/boxscoremiscv3"
# "https://stats.nba.com/stats/boxscorescoringv3"
# "https://stats.nba.com/stats/boxscoreusagev3"
# "https://stats.nba.com/stats/boxscorefourfactorsv3?"
# "https://stats.nba.com/stats/boxscorehustlev2"


#%%
def get_period_params(end_period, range_type, start_period):
    params = {
        "endPeriod": f"{end_period}",
        "endRange": "28800",
        "rangeType": f"{range_type}",
        "startPeriod": f"{start_period}",
        "startRange": "0"
    }

#%%
fl_params = get_period_params(0,0,0)
q1_params = get_period_params(1,1,1)
q2_params = get_period_params(2,1,2)
q3_params = get_period_params(3,1,3)
q4_params = get_period_params(4,1,4)
h1_params = get_period_params(1,1,2)
h1_params = get_period_params(4,1,3)

#%%
url = "https://stats.nba.com/stats/boxscoretraditionalv3"
# Parameters for the request
params = {
    "GameID": "1322300004",
    "LeagueID": "00",
    "endPeriod": "0",
    "endRange": "28800",
    "rangeType": "0",
    "startPeriod": "0",
    "startRange": "0"
}

headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "macOS",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
}

response = requests.get(url, params=params, headers=headers)

#%%
response.status_code

#%%
j = response.json()
j['meta']['request'].split('/')[4]

#%%
import json
json.dump(j, open(BASDIR+'nba_json_box.json', 'w'))

#%%
r = requests.get("http://nba.cloud/games/0022200576", headers=headers)

#%%
r.status_code

#%%
r.text

#%%
url = "https://core-api.nba.com/cp/api/v1.3/feeds/gamecardfeed"
params = {
    "gamedate": "01/05/2000",
    "platform": "web",
}

headers = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "core-api.nba.com",
    "Ocp-Apim-Subscription-Key": "747fa6900c6c4e89a58b81b72f36eb96",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
}

#%%
response = requests.get(url, params=params, headers=headers)

#%%
response.status_code

#%%
r = response.json()
len(r['modules'])

#%%
print(r)

#%%
def parse_games(games:dict[str]) -> dict[str]:
    fmt_games = {}
    i = 0
    for card in games['modules'][0]['cards']:
        print(i)
        i += 1
        data = card['cardData']
        game_id = data['gameId']

        fmt_games[game_id] = {
            'meta':{
                'season_yr':data['seasonYear'],
                'season_type':data['seasonType'],
                'game_time':data['gameTimeEastern']
            },
            'home': data['homeTeam'],
            'away': data['awayTeam']
        }

    return fmt_games

#%%
x = parse_games(r)
print(len(x))

#%%

#%%
r = response.json()

#%%
json.dump(r, open(BASDIR+'nba_json_games.json', 'w'))

#%%




