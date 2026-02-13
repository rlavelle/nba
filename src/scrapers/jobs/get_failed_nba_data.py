import argparse
import configparser
import os
import time

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.scrapers.jobs.get_prev_nba_data import update_nba_data
from src.scrapers.nba.utils.validation import is_date_data_complete
from src.utils.file_io import get_dirs

def collect_data_wth_retry(logger: Logger,
                           retries: int,
                           delay: int,
                           date: str):

    for attempt in range(1, retries + 1):
        try:
            logger.log(f'[ATTEMPT {attempt}]')
            update_nba_data(logger=logger,
                            date=date,
                            skip_scrape=False,
                            skip_insert=False)
            break
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.log(f'[COMPLETE FAILURE]')
                insert_error({'msg': f'complete failure after {retries}: {str(e)}'})

def get_incomplete_games(data_path):
    incomplete_games = []

    folders = get_dirs(data_path)
    for folder in folders:
        path = os.path.join(data_path, folder)

        if int(is_date_data_complete(path, int(folder))) == 0:
            incomplete_games.append(folder)

    return incomplete_games

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA data scraper')
    parser.add_argument('--retries', type=int, help='Total retries for cron job (default: 5)')
    parser.add_argument('--delay', type=int, help='Delay for retries for cron job (default: 10')
    parser.add_argument('--offline', action='store_true', help='Run script offline (no email)')
    args = parser.parse_args()

    if not args.offline:
        logger = Logger(name='incomplete_rerun', daily_cron=True, admin=True)
    else:
        logger = Logger(daily_cron=True, admin=True)

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        data_path = config.get('DATA_PATH', 'games_folder')
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})

        if not args.offline:
            logger.email_log()

        exit()

    try:
        incomplete_games = get_incomplete_games(data_path)
        logger.log(f'[INCOMPLETE GAMES]: {len(incomplete_games)}')
    except Exception as e:
        logger.log(f'[ERROR COLLECTING INCOMPLETE GAMES]: {e}')
        insert_error({'msg': str(e)})

        if not args.offline:
            logger.email_log()

        exit()

    retries = args.retries if args.retries else 5
    delay = args.delay if args.delay else 10

    for game in incomplete_games:
        collect_data_wth_retry(logger=logger,
                               retries=retries,
                               delay=delay,
                               date=game)

    try:
        tmp = get_incomplete_games(data_path)
        logger.log(f'[N RECLAIMED GAMES]: {len(incomplete_games) - len(tmp)}')
    except Exception as e:
        logger.log(f'[ERROR COLLECTING INCOMPLETE GAMES]: {e}')
        insert_error({'msg': str(e)})

    if not args.offline:
        logger.email_log()