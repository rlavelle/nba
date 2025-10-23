#!/usr/bin/env python3
import argparse
import configparser
import datetime
import json
import os
import time

import pandas as pd

from src.config import CONFIG_PATH
from src.db.constants import ODDS_SCHEMAS
from src.db.db_manager import DBManager
from src.db.utils import insert_error, insert_table
from src.logging.logger import Logger
from src.utils.date import date_to_dint, fmt_iso_dint
from src.scrapers.odds.odds_api import OddsApi

# TODO: validate that this works with 100% success rate
#       it should based on the way that team naming conventions work
def J(a:str,b:str):
    A = set(a.lower().split(' '))
    B = set(b.lower().split(' '))

    return len(A.intersection(B)) / len(A.union(B))

def get_upcoming_games(logger: Logger):
    api = OddsApi(logger=logger)
    dbm = DBManager(logger=logger)

    try:
        games = api.get_upcoming_games()
    except Exception as e:
        logger.log(f'[ERROR ON ODDS API UPCOMING GAMES]: {e}')
        insert_error({'msg': str(e)})
        return

    try:
        teams = dbm.get_teams()
    except Exception as e:
        logger.log(f'[ERROR ON READING TEAMS TABLE]: {e}')
        insert_error({'msg': str(e)})
        return

    res = []
    team_mapping = dict()

    for game in games:
        for side in ['home_team', 'away_team']:
            team_name = game[side]
            idx, match = max(
                enumerate(teams.team_name.values),
                key=lambda pair: J(team_name, pair[1])
            )

            db_id = int(teams.loc[idx,'team_id'])

            tmp = {
               'id': game['id'],
               'dint': fmt_iso_dint(game['commence_time']),
               'oods_name': team_name,
               'db_name': match,
               'db_slug': teams.loc[idx,'team_slug'],
               'db_id': db_id,
               'is_home': side == 'home_team'
           }

            team_mapping[game[side]] = db_id
            res.append(tmp)

    return res, team_mapping


def get_spread_ml(logger: Logger, team_mapping: dict[str,str]):
    api = OddsApi(logger=logger)

    try:
        odds = api.get_spread_ml()
    except Exception as e:
        logger.log(f'[ERROR ON SPREAD ML API HIT]: {e}')
        insert_error({'msg': str(e)})
        return

    res_spreads = []
    res_ml = []
    for odd in odds:
        for bookmaker in odd['bookmakers']:
            for market in bookmaker['markets']:
                for outcome in market['outcomes']:
                    tmp = {
                        'dint': fmt_iso_dint(odd['commence_time']),
                        'bookmaker': bookmaker['key'],
                        'last_update': bookmaker['last_update'],
                    }

                    # game_id is matched in post
                    team_id = team_mapping[outcome['name']]
                    if market['key'] == 'h2h':
                        tmp['team_id'] = team_id
                        tmp['price'] = outcome['price']
                        res_ml.append(tmp)

                    elif market['key'] == 'spreads':
                        tmp['team_id'] = team_id
                        tmp['price'] = outcome['price']
                        tmp['point'] = outcome['point']
                        res_spreads.append(tmp)

                    else:
                        logger.log(f'[UNKNOWN MARKET KEY]: {market["key"]}')
                        continue

    return res_spreads, res_ml


def _match_player(player_name, players_df):
    idx, match = max(
        enumerate(players_df.player_name.values),
        key=lambda pair: J(player_name, pair[1])
    )

    player_id = int(players_df.loc[idx, 'player_id'])
    return player_id

def parse_props(logger, id):
    api = OddsApi(logger=logger)
    dbm = DBManager(logger=logger)

    try:
        props = api.get_props(id)
    except Exception as e:
        logger.log(f'[ERROR ON PROPS API HIT]: {e}')
        insert_error({'msg': str(e)})
        return

    try:
        players_df = dbm.get_players()
    except Exception as e:
        logger.log(f'[ERROR ON DBM GET PLAYERS]: {e}')
        insert_error({'msg': str(e)})
        return

    res = []
    for bookmaker in props['bookmakers']:
        market = bookmaker['markets'][0] # we only request 1 market
        for outcome in market['outcomes']:
            tmp = {
                'player_id': _match_player(outcome['description'], players_df),
                'dint': fmt_iso_dint(props['commence_time']),
                'bookmaker': bookmaker['key'],
                'last_update': market['last_update'],
                'odd_type': market['key'],
                'description': outcome['name'],
                'price': outcome['price'],
                'point': outcome['point']
            }
            res.append(tmp)

    return res


def get_props(upcoming_games):
    res_props = []
    games_ids = list(set([g['id'] for g in upcoming_games]))
    for game_id in games_ids:
        tmp = parse_props(logger, game_id)
        res_props.extend(tmp)

    return res_props


def dump_raw_odds(date_path, upcoming_games, res_spreads, res_ml, res_props):
    res = {
        'upcoming': upcoming_games,
        'spreads': res_spreads,
        'ml': res_ml,
        'props': res_props
    }

    os.makedirs(date_path, exist_ok=True)

    i = 0
    while True:
        filename = f"{date_path}.json" if i == 0 else f"{date_path}_{i}.json"
        filepath = os.path.join(date_path, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump(res, f)
            break
        i += 1

def insert_odds_tables(args, res_props, res_spreads, res_ml):
    if args.skip_insert:
        logger.log(f'[SKIP INSERT] Skipping insert for {dint}')
    else:
        names = ['player_props', 'game_spreads', 'game_ml']
        tables = [pd.DataFrame(res_props), pd.DataFrame(res_spreads), pd.DataFrame(res_ml)]

        for table, schema, name in zip(tables, ODDS_SCHEMAS, names):
            try:
                insert_table(table, schema, name, drop=False)
                logger.log(f'[INSERT SUCCESS] {name}')
            except Exception as e:
                logger.log(f'[ERROR ON INSERT - {name}]: {e}')
                insert_error({'msg': str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA odds data collection')
    parser.add_argument('--skip-insert', action='store_true', help='Skip the parse & insert step')
    args = parser.parse_args()

    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True)

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        data_path = config.get('DATA_PATH', 'odds_folder')
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})
        exit()

    date = datetime.date.today()
    dint = date_to_dint(date)
    date_path = os.path.join(data_path, f'{dint}')

    try:
        upcoming_games, team_mapping = get_upcoming_games(logger)
        res_spreads, res_ml = get_spread_ml(logger, team_mapping)
        res_props = get_props(upcoming_games)

        dump_raw_odds(date_path, upcoming_games, res_spreads, res_ml, res_props)

        logger.log(f'[SUCCESS ODDS PULL]: {len(upcoming_games)//2} games collected')

        insert_odds_tables(args, res_props, res_spreads, res_ml)

        end = time.time()
        logger.log(f'[DONE] Runtime: {round((end - start), 2)}s')

    except Exception as e:
        logger.log(f'[ERROR]: {e}')

    logger.email_log()
