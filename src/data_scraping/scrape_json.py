import configparser
from datetime import datetime, timedelta
from src.api.nba_stats_api import NBAStatsApi
import src.api.api_utils as util
import json
from src.config import CONFIG_PATH
from src.logging.logger import Logger


def generate_dates(start_year, end_year):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def date_to_dint(date):
    return int(date.strftime('%Y%m%d'))


def date_to_lookup(date, date_format="%m/%d/%Y"):
    return date.strftime(date_format)


if __name__ == "__main__":
    logger = Logger()
    api = NBAStatsApi()
    dates = generate_dates(2000, 2023)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'path')

    boxscores = config.options('NBA_STATS_ENDPOINTS')
    periods = config.options('TIME_PERIODS')

    for date in dates:
        lookup = date_to_lookup(date)
        dint = date_to_dint(date)

        games = api.get_games(date=lookup)

        if 'error' in games:
            print(f'bad api hit on {date}')
            logger.log(f'bad api hit on {date}')
            continue

        if len(games) == 0:
            continue

        fmt_games = util.parse_games(games=games)
        print(f'{dint} {len(fmt_games)} games')

        json.dump(fmt_games, open(f'{data_path}/games/{dint}_games.json', 'w'))

        print('collecting games...')
        for game_id in list(fmt_games.keys()):
            score = api.get_boxscore(game_id=game_id, endpoint='boxscore_traditional', period='full_game')

            if 'error' in score:
                print(f'bad api hit on {game_id}')
                logger.log(f'bad api hit on {game_id}')
                continue

            validate_game = score['meta']['request'].split('/')[4]
            assert game_id == validate_game, f'{game_id} <> {validate_game}'
            fmt_game = util.parse_boxscore(score=score)

            json.dump(fmt_game, open(f'{data_path}/stats/{game_id}_stats.json', 'w'))
