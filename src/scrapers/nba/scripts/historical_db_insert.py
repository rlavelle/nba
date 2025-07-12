import configparser

import pandas as pd
import os
import numpy as np

from src.config import CONFIG_PATH
from src.db.utils import insert_table
from src.scrapers.nba.utils import time_to_minutes, get_dirs, parse_dumped_game_data, clean_tables
from src.scrapers.nba.constants import PLAYER_DUPE_COLS, GAME_DUPE_COLS
from src.db.schema import TEAMS_SCHEMA, PLAYERS_SCHEMA, GAMES_META_SCHEMA, \
    PLAYER_STATS_SCHEMA, GAME_STATS_SCHEMA


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db = config.get('DB_PATHS', 'db_path')
    games_folder = config.get('DATA_PATH', 'games_folder')

    seen_players = set()
    seen_teams = set()

    master_player_meta = []
    master_player_stats = []
    master_team_meta = []
    master_game_meta = []
    master_game_stats = []

    games_by_date = sorted(os.listdir(games_folder))

    i = 0
    for date in games_by_date:
        games = get_dirs(os.path.join(games_folder, date))
        for game in games:
            path = os.path.join(games_folder, date, game)
            game_meta, game_stats, player_data, player_stats, team_data = parse_dumped_game_data(path, int(date), game)

            for pdata in player_data:
                if not pdata['player_id'] in seen_players:
                    master_player_meta.append(pdata)
                    seen_players.add(pdata['player_id'])

            for tdata in team_data:
                if not tdata['team_id'] in seen_teams:
                    master_team_meta.append(tdata)
                    seen_teams.add(tdata['team_id'])

            master_player_stats.extend(player_stats)
            master_game_stats.extend(game_stats)
            master_game_meta.append(game_meta)
        i += 1
        if i > 100:
            break

    game_meta_table, game_stats_table, team_meta_table, player_meta_table, player_stats_table = clean_tables(
        master_game_meta, master_game_stats, master_team_meta, master_player_meta, master_player_stats
    )

    tables = [game_meta_table, team_meta_table, player_meta_table, player_stats_table, game_stats_table]
    schemas = [GAMES_META_SCHEMA, TEAMS_SCHEMA, PLAYERS_SCHEMA, PLAYER_STATS_SCHEMA, GAME_STATS_SCHEMA]
    names = ['games', 'teams', 'players', 'player_stats', 'game_stats']
    for table, schema, name in zip(tables, schemas, names):
        insert_table(table, schema, name, db, drop=True)
