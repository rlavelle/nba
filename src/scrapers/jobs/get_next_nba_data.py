import argparse
import configparser
import datetime
import json
import os
import pickle
import time

from src.config import CONFIG_PATH
from src.db.db_manager import DBManager
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.scrapers.nba.utils.date import date_to_dint
from src.scrapers.odds.odds_api import OddsApi

# TODO: validate that this works with 100% success rate
#       it should based on the way that team naming conventions work
def J(a:str,b:str):
    A = set(a.lower().split(' '))
    B = set(b.lower().split(" "))
    return len(A ^ B) / len(A | B)

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

            db_id = teams.loc[idx,'team_id'].values[0]

            tmp = {
               'id': game['id'],
               'date': game['commence_time'],
               'oods_name': team_name,
               'db_name': match,
               'db_slug': teams.loc[idx,'team_slug'].values[0],
               'db_id': db_id,
               'is_home': side == 'home_team'
           }

            team_mapping[game[side]] = db_id
            res.append(tmp)

    return res, team_mapping


def get_spread_ml(logger: Logger, team_mapping: dict[str,str]):
    api = OddsApi(logger=logger)
    dbm = DBManager(logger=logger)

    try:
        odds = api.get_spread_ml()
    except Exception as e:
        logger.log(f'[ERROR ON SPREAD ML API HIT]: {e}')
        insert_error({'msg': str(e)})
        return

    try:
        players = dbm.get_players()
    except Exception as e:
        logger.log(f'[ERROR ON READING TEAMS TABLE]: {e}')
        insert_error({'msg': str(e)})
        return

    res_spreads = []
    res_ml = []
    for odd in odds:
        for bookmaker in odd['bookmakers']:
            for market in bookmaker['markets']:
                for outcome in market['outcomes']:
                    tmp = {
                        'id': odd['id'],
                        'date': odd['commence_time'],
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
                        tmp['team_name'] = team_id
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

    player_id = players_df.loc[idx, 'player_id'].values[0]
    return player_id

def parse_props(logger, id):
    api = OddsApi(logger=logger)

    try:
        props = api.get_props(id)
    except Exception as e:
        logger.log(f'[ERROR ON PROPS API HIT]: {e}')
        insert_error({'msg': str(e)})
        return

    res = []
    for bookmaker in props['bookmakers']:
        market = bookmaker['markets'][0] # we only request 1 market
        for outcome in market['outcomes']:
            tmp = {
                'game_id': id,
                'player_id': _match_player(outcome['description']),
                'date': bookmaker['commence_time'],
                'bookmaker': bookmaker['key'],
                'last_update': market['last_update'],
                'odd_type': market['key'],
                'desc': outcome['name'],
                'price': outcome['price'],
                'point': outcome['point']
            }
            res.append(tmp)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA odds API')
    parser.add_argument('--retries', type=int, help='Total retries for cron job (default: 5)')
    parser.add_argument('--delay', type=int, help='Delay for retries for cron job (default: 10')
    args = parser.parse_args()

    retries = args.retries if args.retries else 5
    delay = args.delay if args.delay else 10

    logger = Logger(fpath='cron_path')

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'odds_folder')
    date = datetime.date.today()
    dint = date_to_dint(date)
    date_path = os.path.join(data_path, f'{dint}')

    for attempt in range(1, retries + 1):
        try:
            logger.log(f'[ATTEMPT {attempt}]')
            upcoming_games, team_mapping = get_upcoming_games(logger)
            res_spreads, res_ml = get_spread_ml(logger, team_mapping)

            res_props = []
            for game in upcoming_games:
                tmp = parse_props(logger, team_mapping, game['id'])
                res_props.extend(tmp)

            res = {
                'upcoming': upcoming_games,
                'spreads': res_spreads,
                'ml': res_ml,
                'props': res_props
            }
            
            # TODO: temp dumping until DB is setup
            json.dump(res, open(os.path.join(date_path, f'odds_dump.json'), 'w'))

            break
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.log(f'[COMPLETE FAILURE]')
                insert_error({'msg': f'complete failure after {retries}: {str(e)}'})
