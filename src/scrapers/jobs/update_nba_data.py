#!/usr/bin/env python3

import configparser
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.config import CONFIG_PATH
from src.logging.logger import Logger
from datetime import datetime, timedelta

from src.scrapers.nba.utils import fetch_and_save_boxscore, parse_games, date_to_dint, date_to_lookup, get_dirs, \
    insert_parsed_data_by_day, is_date_data_complete, is_game_data_complete, is_bad_game, is_empty_game


def update_nba_data():
    start = time.time()

    logger = Logger(fpath='cron_path')
    api = NBAStatsApi(logger=logger)
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    data_path = config.get('DATA_PATH', 'games_folder')
    boxscores = config.options('NBA_STATS_ENDPOINTS')

    date  = datetime.today()
    lookup = date_to_lookup(date)
    dint = date_to_dint(date)

    logger.log(f'[SCRAPE FOR {date}]')

    date_path = os.path.join(data_path, str(dint))
    if is_date_data_complete(date_path, dint):
        logger.log(f'skipping {dint}... data pulled')
    else:
        games = api.get_games(date=lookup)

        if 'error' in games:
            print(f'bad api hit on {date}')
            logger.log(f'bad api hit on {date}')
            logger.log(games['error'])
            return

        if len(games) == 0:
            logger.log(f'No games on {date}')
            return

        fmt_games = parse_games(games=games)
        print(f'{dint} {len(fmt_games)} games')

        os.makedirs(date_path, exist_ok=True)
        json.dump(fmt_games, open(os.path.join(date_path, f'{dint}_games.json'), 'w'))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for game_id, v in fmt_games.items():
                game_path = os.path.join(date_path, f'{game_id}')

                if is_bad_game(v):
                    logger.log(f'[ERROR IN GAME]: {game_id}\n{v}')
                    continue

                if is_empty_game(v):
                    logger.log(f'[EMPTY GAME]: {game_id}\n{v}')
                    continue

                if is_game_data_complete(game_path):
                    logger.log(f'skipping {dint}-{game_id}... data pulled')
                    continue

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
                future.result()

        end = time.time()
        logger.log(f'[SCRAPE FOR {date} FINISHED] {len(futures)} completed... {round(end - start, 2) / 60} min')

    if is_date_data_complete(date_path, dint):
        try:
            insert_parsed_data_by_day(date_path, str(dint), logger)
            logger.log(f'[INSERT SUCCESS]')
        except Exception as e:
            logger.log(f'[ERROR ON PARSE]: {e}')

    else:
        logger.log(f'[ERROR IN SCRAPE]: Not all data collected, check logs')


if __name__ == "__main__":
    update_nba_data()