import configparser
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scrapers.nba.nba_stats_api import NBAStatsApi
import src.scrapers.nba.utils as util
from src.config import CONFIG_PATH
from src.logging.logger import Logger
from src.scrapers.nba.utils import is_date_data_complete, is_game_data_complete

if __name__ == "__main__":
    logger = Logger()
    api = NBAStatsApi()
    dates = util.generate_dates(2014, 7, 1)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'games_folder')

    boxscores = config.options('NBA_STATS_ENDPOINTS')
    periods = config.options('TIME_PERIODS')

    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for date in dates:
            lookup = util.date_to_lookup(date)
            dint = util.date_to_dint(date)

            date_path = os.path.join(data_path, f'{dint}')
            if is_date_data_complete(date_path, dint):
                logger.log(f'skipping {dint}... data pulled')
                continue

            games = api.get_games(date=lookup)

            if 'error' in games:
                print(f'bad api hit on {date}')
                logger.log(f'bad api hit on {date}')
                continue

            if len(games) == 0:
                continue

            fmt_games = util.parse_games(games=games)
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
                    futures.append(
                        executor.submit(
                            util.fetch_and_save_boxscore,
                            game_id, boxscore, api, util, game_path, logger
                        )
                    )

        print(f'Submitted {len(futures)} tasks... waiting for completion.')

        for future in as_completed(futures):
            future.result()