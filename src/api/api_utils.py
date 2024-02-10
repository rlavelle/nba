import requests
import time

from src.logging.logger import Logger


def parse_boxscore(score:dict[str]) -> dict[str]:
    metric = list(score.keys())[1]
    return score[metric]


def parse_games(games:dict[str]) -> dict[str]:
    fmt_games = {}
    for card in games['modules'][0]['cards']:
        data = card['cardData']
        game_id = data['gameId']

        fmt_games[game_id] = {
            'meta':{
                'season_yr':data['seasonYear'],
                'season_type':data['seasonType'],
                'game_time':data['gameTimeEastern']
            },
            'home': data['homeTeam'],
            'away': data['awayTeam']
        }

    return fmt_games


def post_with_retry(url: str, params: dict[str], headers: dict[str], max_retries=10, retry_interval=10) -> dict[str]:
    logger = Logger()

    for attempt in range(1, max_retries + 1):
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 503:
            msg = f"Attempt {attempt}: Service Unavailable (503). Retrying in {retry_interval} seconds."
            print(msg)
            logger.log(message=msg)
            logger.log(message=f'{url} {params}')

            time.sleep(retry_interval)
        elif response.status_code != 200:
            msg = f"Attempt {attempt}: Unexpected status code {response.status_code}. Retrying in {retry_interval} seconds."
            print(msg)
            logger.log(message=msg)
            logger.log(message=f'{url} {params}')

            time.sleep(retry_interval)
        else:
            return response.json()

    msg = f"Failed after {max_retries} attempts. Returning empty dictionary. {url} {params}"
    print(msg)
    logger.log(message=msg)
    logger.log(message=f'{url} {params}')

    return {'error': response}


def post(url:str, params:dict[str], headers:dict[str]) -> dict[str]:
    return post_with_retry(url=url, params=params, headers=headers)
