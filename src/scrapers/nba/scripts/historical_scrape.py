#!/usr/bin/env python3

import configparser
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.config import CONFIG_PATH
from src.logging.logger import Logger
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.scrapers.nba.utils.api_utils import fetch_and_save_boxscore
from src.scrapers.nba.utils.parsing import parse_games
from src.scrapers.nba.utils.validation import is_date_data_complete, is_game_data_complete
from src.utils.date import generate_dates, date_to_dint, date_to_lookup

if __name__ == "__main__":
    logger = Logger()
    api = NBAStatsApi()
    dates = generate_dates(datetime(2019, 7, 1))

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'games_folder')

    boxscores = config.options('NBA_STATS_ENDPOINTS')
    periods = config.options('TIME_PERIODS')

    raw_errors = open('/Users/rowanlavelle/Documents/Projects/nba/logs/errors.txt', 'r').readlines()
    # (game_id, boxscore)
    errors = list(set([tuple(x.split()) for x in raw_errors]))

    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for date in dates:
            lookup = date_to_lookup(date)
            dint = date_to_dint(date)

            date_path = os.path.join(data_path, f'{dint}')
            if is_date_data_complete(date_path, dint):
                logger.log(f'skipping {dint}... data pulled')
                continue

            games = api.get_games(date=lookup)

            if 'error' in games:
                print(f'bad api hit on {date}')
                logger.log(f'bad api hit on {date}')
                logger.log(games['error'])
                continue

            if len(games) == 0:
                continue

            fmt_games = parse_games(games=games)
            print(f'{dint} {len(fmt_games)} games')

            os.makedirs(date_path, exist_ok=True)
            json.dump(fmt_games, open(os.path.join(date_path, f'{dint}_games.json'), 'w'))

            for game_id, v in fmt_games.items():
                game_path = os.path.join(date_path, f'{game_id}')
                game_file = os.path.join(game_path, f'{game_id}_meta.json')

                if is_game_data_complete(game_path):
                    logger.log(f'skipping {dint}-{game_id}... data pulled')
                    continue

                os.makedirs(game_path, exist_ok=True)
                json.dump(v, open(game_file, 'w'))

                for boxscore in boxscores:
                    err = False
                    for a, b in errors:
                        if a == game_id and b == boxscore:
                            print('SKIPPING ERROR FILE')
                            err = True
                    if err:
                        continue

                    futures.append(
                        executor.submit(
                            fetch_and_save_boxscore,
                            game_id, boxscore, api, game_path, logger
                        )
                    )

        print(f'Submitted {len(futures)} tasks... waiting for completion.')

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.log(f'[EXCEPTION ON FUTURE]: {e}')
