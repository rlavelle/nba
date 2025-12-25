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

if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True, admin=True)
    logger.log(f'[STARTING PREDICTIONS]')

    parser = argparse.ArgumentParser(description='NBA prediction script')
    parser.add_argument('--date', type=str, help='Date to pull in YYYY-MM-DD format (default: today)')
    parser.add_argument('--offline', action='store_true', help='offline testing')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.date.today()
    curr_date = date_to_dint(date)

    dbm = DBManager(logger=logger)

    try:
        prop_results = dbm.get_prop_results()
    except Exception as e:
        logger.log(f'[ERROR LOADING PROP RESULTS]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        ml_results = dbm.get_money_line_results()
    except Exception as e:
        logger.log(f'[ERROR LOADING MONEYLINE RESULTS]: {e}')
        insert_error({'msg': str(e)})
        exit()

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

    games['team_id'] = games.team_id.astype(int)
    ml_results['team_id'] = ml_results.team_id.astype(int)
    game_data = pd.merge(ml_results, games,
                         on=['team_id', 'dint'],
                         how='left')





