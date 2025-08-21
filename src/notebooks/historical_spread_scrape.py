#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:44:09 2025

@author: rowanlavelle
"""

#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#%%
BASE_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{}/"

# Seasons 2016–17 through 2022–23
SEASONS = ["{}-{}".format(y, str(y+1)[2:]) for y in range(2016, 2023)]

def scrape_season(season: str) -> pd.DataFrame:
    """
    Scrape a single season's NBA odds table into a DataFrame.
    """
    url = BASE_URL.format(season)
    print(f"Fetching {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/92.0.4515.131 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="table bg-white table-hover table-bordered table-sm")
    if not table:
        print(f"[{season}] No table found.")
        return pd.DataFrame()

    rows = []
    for row in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if cols:
            rows.append(cols)

    # First row is header
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    df["season"] = season
    return df

#%%
all_dfs = []
for season in SEASONS:
    df = scrape_season(season)
    if not df.empty:
        all_dfs.append(df)
    time.sleep(1)  # polite delay

df_all = pd.concat(all_dfs, ignore_index=True)
print(df_all.head())
print(f"Scraped {len(df_all)} rows across {len(SEASONS)} seasons.")

#%%
# Optional: save to CSV
df_all.to_pickle("data/tmp/nba_odds_archive_23.pkl")

#%%
df_all = pd.read_pickle('data/tmp/nba_odds_archive_23.pkl')

def build_game_df(df_all: pd.DataFrame) -> pd.DataFrame:
    visitors = df_all[df_all["VH"] == "V"].reset_index(drop=True)
    homes = df_all[df_all["VH"] == "H"].reset_index(drop=True)

    # Merge by index (assumes alternating rows)
    df_games = pd.concat([visitors.add_suffix("_away"), homes.add_suffix("_home")], axis=1)

    # Convert numeric columns
    num_cols = ["1st", "2nd", "3rd", "4th", "Final", "Open", "Close", "ML", "2H"]
    for col in num_cols:
        for suffix in ["_home", "_away"]:
            new_col = col + suffix
            df_games[new_col] = pd.to_numeric(df_games[new_col], errors="coerce")


    df_games["ml_home"] = df_games["ML_home"]
    df_games["ml_away"] = df_games["ML_away"]
    
    # Actual total and spread
    df_games["actual_total"] = df_games["Final_home"] + df_games["Final_away"]
    def compute_actual_spread(row):
        if row["ML_home"] < row["ML_away"]:  # home is favorite
            return row["Final_home"] - row["Final_away"]
        else:  # away is favorite
            return row["Final_away"] - row["Final_home"]
    
    df_games["actual_spread"] = df_games.apply(compute_actual_spread, axis=1)
    # Determine open/close totals
    # Usually the spread is on the favored team, total is same for both rows
    # We'll take spread from the team that has numeric Close
    def get_spread_total(row, col_type="Close"):
        home = row[f"{col_type}_home"]
        away = row[f"{col_type}_away"]
    
        # Determine spread: take the number < 100
        spread = home if (not pd.isna(home) and home < 100) else away
        # Determine total: take the number >= 100
        total = home if (not pd.isna(home) and home >= 100) else away
    
        return pd.Series({f"{col_type.lower()}_spread": spread, f"{col_type.lower()}_total": total})

    df_games[["close_spread", "close_total"]] = df_games.apply(lambda r: get_spread_total(r, "Close"), axis=1)
    df_games[["open_spread", "open_total"]] = df_games.apply(lambda r: get_spread_total(r, "Open"), axis=1)

    # Build final DataFrame
    df_final = df_games[[
        "season_home", "Date_home", "Team_home", "Team_away",
        "actual_total", "actual_spread",
        "open_total", "close_total", "open_spread", "close_spread",
        "ml_home", "ml_away"
    ]].copy()

    df_final = df_final.rename(columns={
        "season_home": "season",
        "Date_home": "date",
        "Team_home": "home_team",
        "Team_away": "away_team"
    })

    return df_final
#%%
df = build_game_df(df_all)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib auto

#%%
x = df[(df.close_spread < 100) & (df.open_spread < 100)].copy()

#%%
tmp = df[(df.home_team == 'Boston') | (df.away_team=='Boston')]
tmp = tmp[(tmp.close_spread < 100) & (tmp.open_spread < 100)].copy()

#%%
plt.clf()
plt.scatter(np.arange(x.shape[0]), x.actual_spread.abs())
plt.scatter(np.arange(x.shape[0]), x.close_spread.abs())
plt.show()

#%%
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

mae(x.actual_spread.abs(), x.open_spread)

#%%






