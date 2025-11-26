#!/usr/bin/env python3
import argparse
import configparser
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from src.db.constants import SCHEMAS
from src.db.utils import insert_error, insert_table
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.config import CONFIG_PATH
from src.logging.logger import Logger
from datetime import datetime

from src.scrapers.nba.utils.api_utils import fetch_and_save_boxscore
from src.scrapers.nba.utils.validation import is_date_data_complete, is_game_data_complete, is_empty_game, is_bad_game
from src.utils.date import date_to_dint, date_to_lookup
from src.scrapers.nba.utils.parsing import parse_games, parse_dumped_data_by_day

"""
This job is run to scrape games *after* they occur
    - runs in the morning to get results of prev day
"""

def get_configured_data_path(config_path: str) -> Tuple[str, list[str]]:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config.get('DATA_PATH', 'games_folder'), config.options('NBA_STATS_ENDPOINTS')


def scrape_games_for_day(date: datetime, data_path: str, api: NBAStatsApi, logger: Logger, boxscores: list[str]) -> None:
    lookup = date_to_lookup(date)
    dint = date_to_dint(date)
    logger.log(f'[SCRAPE FOR {date}]')

    date_path = os.path.join(data_path, str(dint))
    os.makedirs(date_path, exist_ok=True)

    games = api.get_games(date=lookup)

    if 'error' in games:
        msg = f'Bad API hit on {date}: {games["error"]}'
        logger.log(msg)
        insert_error({'msg': msg})
        return

    if len(games) == 0:
        logger.log(f'No games on {date}')
        return

    fmt_games = parse_games(games)
    json.dump(fmt_games, open(os.path.join(date_path, f'{dint}_games.json'), 'w'))
    logger.log(f'{dint}: {len(fmt_games)} games found')

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for game_id, game_meta in fmt_games.items():
            game_path = os.path.join(date_path, f'{game_id}')
            os.makedirs(game_path, exist_ok=True)

            if is_bad_game(game_meta):
                logger.log(f'[BAD GAME SKIPPED]: {game_id}')
                insert_error({'game_id': game_id, 'msg': f'[BAD GAME SKIPPED]: {game_id}'})
                continue

            if is_empty_game(game_meta):
                logger.log(f'[EMPTY GAME]: {game_id}')
                continue

            if is_game_data_complete(game_path):
                logger.log(f'skipping {dint}-{game_id}... data already present')
                continue

            json.dump(game_meta, open(os.path.join(game_path, f'{game_id}_meta.json'), 'w'))

            for boxscore in boxscores:
                future = executor.submit(fetch_and_save_boxscore, game_id, boxscore, api, game_path, logger)
                futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.log(f'[THREAD ERROR]: {str(e)}')
                insert_error({'msg': str(e)})


def parse_and_insert_if_complete(date: datetime, data_path: str, logger: Logger) -> None:
    dint = date_to_dint(date)
    date_path = os.path.join(data_path, str(dint))

    if not is_date_data_complete(date_path, dint):
        logger.log(f'[ERROR IN SCRAPE]: Not all data collected for {dint}')
        if len(os.listdir(date_path)) == 1:
            logger.log(f'[EMPTY GAME DATA]: Check {date_path}/{dint}_games.json')
            return
        insert_error({'msg': f'Data incomplete for {dint}'})
        return

    try:
        tables = parse_dumped_data_by_day(date_path, str(dint), logger)
        names = ['games', 'teams', 'players', 'player_stats', 'game_stats']
    except Exception as e:
        logger.log(f'[ERROR ON PARSE]: {e}')
        insert_error({'msg': str(e)})
        return

    for table, schema, name in zip(tables, SCHEMAS, names):
        try:
            insert_table(table, schema, name, drop=False)
            logger.log(f'[INSERT SUCCESS] {name}')
        except Exception as e:
            logger.log(f'[ERROR ON INSERT - {name}]: {e}')
            insert_error({'msg': str(e)})


def update_nba_data(logger: Logger, args):
    start = time.time()

    date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.today()
    dint = int(date.strftime('%Y%m%d'))

    api = NBAStatsApi(logger=logger)

    try:
        data_path, boxscores = get_configured_data_path(CONFIG_PATH)
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})
        return

    date_path = os.path.join(data_path, str(dint))

    if args.skip_scrape:
        logger.log(f'[SKIP SCRAPE] Skipping scrape step for {dint}')
    elif is_date_data_complete(date_path, dint):
        logger.log(f'[SKIP] Data already complete for {dint}')
    else:
        scrape_games_for_day(date, data_path, api, logger, boxscores)

    if args.skip_insert:
        logger.log(f'[SKIP INSERT] Skipping parse & insert for {dint}')
    else:
        parse_and_insert_if_complete(date, data_path, logger)

    end = time.time()
    logger.log(f'[DONE] Runtime: {round((end - start)/60, 2)} minutes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA data scraper')
    parser.add_argument('--retries', type=int, help='Total retries for cron job (default: 5)')
    parser.add_argument('--delay', type=int, help='Delay for retries for cron job (default: 10')
    parser.add_argument('--date', type=str, help='Date to scrape in YYYY-MM-DD format (default: today)')
    parser.add_argument('--skip-scrape', action='store_true', help='Skip the scraping step')
    parser.add_argument('--skip-insert', action='store_true', help='Skip the parse & insert step')
    args = parser.parse_args()

    retries = args.retries if args.retries else 5
    delay = args.delay if args.delay else 10

    logger = Logger(fpath='cron_path', daily_cron=True, admin=True)

    for attempt in range(1, retries + 1):
        try:
            logger.log(f'[ATTEMPT {attempt}]')
            update_nba_data(logger, args)
            break
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.log(f'[COMPLETE FAILURE]')
                insert_error({'msg': f'complete failure after {retries}: {str(e)}'})

    logger.email_log()
