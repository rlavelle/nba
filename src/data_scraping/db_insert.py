import configparser

import pandas as pd
import json
from sqlalchemy import create_engine, text
import os
import numpy as np
from datetime import datetime

from src.config import CONFIG_PATH
from src.types.db_types import GAMES_SCHEMA, TEAMS_SCHEMA, PLAYERS_SCHEMA, STATS_SCHEMA

SEASON_TYPE_MAP = {
    'Regular Season': '00',
    'Playoffs': '01',
    'All-Star': '02',
    'Preseason': '03',
    'Summer League': '04',
    'PlayIn': '05',
    'IST Championship': '06'
}


def get_game_data(game, date, game_id):
    return {
        'game_id': game_id,
        'season': game['meta']['season_yr'],
        'season_type': game['meta']['season_type'],
        'season_type_code': SEASON_TYPE_MAP[game['meta']['season_type']],
        'dint': date,
        'date': datetime.strptime(str(date), '%Y%m%d'),
        'home_team': game['home']['teamId'],
        'away_team': game['away']['teamId'],
        'home_score': game['home']['score'],
        'away_score': game['away']['score']
    }


def get_team_data(team):
    return {
        'team_id': team['teamId'],
        'team_name': team['teamName'],
        'team_slug': team['teamTricode']
    }


def get_player_data(player):
    return {
        'player_id': player['personId'],
        'player_name': player['firstName'] + ' ' + player['familyName'],
        'player_slug': player['playerSlug']
    }


def get_stats_data(stats, game_id, player_id, team_id, position):
    return {
        'player_id': player_id,
        'team_id': team_id,
        'game_id': game_id,
        'position': position,
        **stats
    }


def bad_game(game):
    if 'home' in game and game['home'] and 'away' in game and game['away']:
        return False
    return True


def time_to_minutes(time_string):
    x = list(map(int, time_string.split(':')))
    if len(x) == 2:
        total_seconds = min(x[0] * 60 + x[1], 48 * 60)
    else:
        total_seconds = min(abs(x[0]) * 60, 48 * 60)

    return total_seconds / 60


def insert_table(table, schema, name, db):
    engine = create_engine(f'sqlite:///{db}')

    with engine.connect() as connection:
        statements = [statement.strip() for statement in schema.split(';') if statement.strip()]

        # Execute each statement separately
        for statement in statements:
            connection.execute(text(statement))

    table.to_sql(name, con=engine, index=False, if_exists='append')


def insert_game_team_tables(folder, db, verbose=False):
    # todo: teams change names, and cities, and its a fucking mess
    #   re: the hornets and the pelicans fuckery
    files = os.listdir(folder)

    seen_teams = {}

    games = []
    teams = []
    for fname in files:
        fpath = os.path.join(folder + fname)
        if not os.path.isfile(fpath):
            continue

        j = json.load(open(fpath, 'r'))
        date = fpath.split('_')[0].split('/')[-1]

        for game_id in list(j.keys()):

            game = j[game_id]

            if bad_game(game):
                continue

            gdata = get_game_data(game, date, game_id)
            hdata = get_team_data(game['home'])
            adata = get_team_data(game['away'])

            if hdata['team_id'] in seen_teams:
                if hdata['team_slug'] != seen_teams[hdata['team_id']]['team_slug'] and \
                        gdata['date'] > seen_teams[hdata['team_id']]['date']:

                    if verbose:
                        print(f'team id: {hdata["team_id"]} switched from {seen_teams[hdata["team_id"]]["team_slug"]} to'
                              f' {hdata["team_slug"]} on {gdata["date"]}')

                    seen_teams[hdata['team_id']]['team_slug'] = hdata['team_slug']
                    seen_teams[hdata['team_id']]['team_name'] = hdata['team_name']
                    seen_teams[hdata['team_id']]['date'] = gdata['date']
            else:
                seen_teams[hdata['team_id']] = {}
                seen_teams[hdata['team_id']]['team_slug'] = hdata['team_slug']
                seen_teams[hdata['team_id']]['team_name'] = hdata['team_name']
                seen_teams[hdata['team_id']]['date'] = gdata['date']

            if adata['team_id'] in seen_teams:
                if adata['team_slug'] != seen_teams[adata['team_id']]['team_slug'] and \
                        gdata['date'] > seen_teams[adata['team_id']]['date']:

                    if verbose:
                        print(f'team id: {adata["team_id"]} switched from {seen_teams[adata["team_id"]]["team_slug"]} to '
                              f'{adata["team_slug"]} on {gdata["date"]}')

                seen_teams[adata['team_id']]['team_slug'] = adata['team_slug']
                seen_teams[adata['team_id']]['team_name'] = adata['team_name']
                seen_teams[adata['team_id']]['date'] = gdata['date']
            else:
                seen_teams[adata['team_id']] = {}
                seen_teams[adata['team_id']]['team_slug'] = adata['team_slug']
                seen_teams[adata['team_id']]['team_name'] = adata['team_name']
                seen_teams[adata['team_id']]['date'] = gdata['date']

            games.append(gdata)
            teams.extend([hdata, adata])

    game_table = pd.DataFrame(games).drop_duplicates()
    team_table = pd.DataFrame(teams).drop_duplicates(subset=['team_id'])

    team_table['team_name'] = team_table.team_id.apply(lambda id: seen_teams[id]['team_name'])
    team_table['team_slug'] = team_table.team_id.apply(lambda id: seen_teams[id]['team_slug'])

    game_table = game_table.replace('', np.nan)
    team_table = team_table.replace('', np.nan)

    insert_table(game_table, GAMES_SCHEMA, 'games', db)
    insert_table(team_table, TEAMS_SCHEMA, 'teams', db)


def insert_player_stats_tables(folder, db, verbose=False):
    files = os.listdir(folder)

    players = []
    stats = []
    for fname in files:
        fpath = os.path.join(folder + fname)
        if not os.path.isfile(fpath):
            continue

        j = json.load(open(fpath, 'r'))
        game_id = j['gameId']

        hteam = j['homeTeam']
        ateam = j['awayTeam']
        teams = [hteam, ateam]

        for team in teams:
            team_id = team['teamId']
            team_players = team['players']

            for player in team_players:
                pdata = get_player_data(player)

                if 'statistics' not in player:
                    continue

                sdata = get_stats_data(player['statistics'], game_id, pdata['player_id'], team_id, player['position'])
                players.append(pdata)
                stats.append(sdata)

    player_table = pd.DataFrame(players).drop_duplicates(subset='player_id', keep='first')
    stats_table = pd.DataFrame(stats).drop_duplicates()

    player_table = player_table.replace('', np.nan)
    stats_table = stats_table.replace('', np.nan)

    stats_table = stats_table[~stats_table.minutes.isna()]
    stats_table['minutes'] = stats_table.minutes.apply(time_to_minutes)
    stats_table = stats_table[stats_table.minutes > 0]

    insert_table(player_table, PLAYERS_SCHEMA, 'players', db)
    insert_table(stats_table, STATS_SCHEMA, 'stats', db)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db = config.get('DB_PATHS', 'db_path')
    games_folder = config.get('DATA_PATH', 'games_folder')
    stats_folder = config.get('DATA_PATH', 'stats_folder')

    insert_game_team_tables(games_folder, db)
    print('game table / team table inserted')
    insert_player_stats_tables(stats_folder, db)
    print('player table / stats table inserted')