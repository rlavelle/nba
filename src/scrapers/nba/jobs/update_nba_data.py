#!/usr/bin/env python3

import configparser
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.config import CONFIG_PATH
from src.logging.logger import Logger
from datetime import datetime

from src.scrapers.nba.utils import fetch_and_save_boxscore, parse_games, date_to_dint, date_to_lookup, get_dirs, \
    insert_parsed_data_by_day, is_date_data_complete

if __name__ == "__main__":
    logger = Logger(fpath='cron_path')
    api = NBAStatsApi()

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'games_folder')

    boxscores = config.options('NBA_STATS_ENDPOINTS')
    periods = config.options('TIME_PERIODS')

    date = datetime.today()
    logger.log(f'[SCRAPE FOR {date}]')
    lookup = date_to_lookup(date)
    dint = date_to_dint(date)

    games = api.get_games(date=lookup)

    if 'error' in games:
        print(f'bad api hit on {date}')
        logger.log(f'bad api hit on {date}')

    if len(games) > 0:
        fmt_games = parse_games(games=games)
        print(f'{dint} {len(fmt_games)} games')

        date_path = os.path.join(data_path, str(dint))
        os.makedirs(date_path, exist_ok=True)
        json.dump(fmt_games, open(os.path.join(date_path, f'{dint}_games.json'), 'w'))

        with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for game_id, v in fmt_games.items():
                    game_path = os.path.join(date_path, f'{game_id}')
                    os.makedirs(game_path, exist_ok=True)
                    json.dump(v, open(os.path.join(game_path, f'{game_id}_meta.json'), 'w'))

                    for boxscore in boxscores:
                        futures.append(
                            executor.submit(
                                fetch_and_save_boxscore,
                                game_id, boxscore, api, game_path, logger
                            )
                        )

                for future in as_completed(futures):
                    future.result()  # Propagate exceptions if any

        if is_date_data_complete(date_path, dint):
            db = config.get('DB_PATHS', 'db_path')
            insert_parsed_data_by_day(date_path, db, str(dint))
        else:
            logger.log(f'[INSERT ERROR] {dint} data not complete')

    else:
        logger.log(f'No games on {date}')