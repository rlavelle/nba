import configparser
import json
import random
import time

import requests

from src.config import CONFIG_PATH
from src.logging.logger import Logger


class NBAStatsApi:
    def __init__(self, logger=None):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)

        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

    def get_boxscore(self, game_id:str, endpoint:str, period:str) -> dict[str]:
        assert endpoint in self.config.options('NBA_STATS_ENDPOINTS'), f'{endpoint} not in valid endpoints'
        assert period in self.config.options('TIME_PERIODS'), f'{period} not in valid time periods'

        url = self.config.get('NBA_STATS_ENDPOINTS', endpoint)
        headers = json.loads(self.config.get('NBA_STATS_HEADERS', 'headers'))
        params = json.loads(self.config.get('TIME_PERIODS', period))
        params['GameID'] = game_id

        response = self._get(url=url, params=params, headers=headers)
        return response

    def get_games(self, date:str) -> dict[str]:
        url = self.config.get('NBA_GAMES_ENDPOINTS', 'nba_games')
        headers = json.loads(self.config.get('NBA_GAMES_HEADERS', 'headers'))
        params = {
            'gamedate': date,
            'platform': 'web'
        }
        response = self._get(url=url, params=params, headers=headers)

        if 'error' in response:
            return response

        if len(response['modules']) == 0:
            return {}

        return response

    def _get_retry(self,
                   url: str,
                   params: dict[str],
                   headers: dict[str],
                   max_retries=10,
                   retry_interval=10,
                   timeout=10) -> dict[str]:

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)

                if response.status_code == 200:
                    return response.json()

                # Handle known retryable status codes
                if response.status_code == 503 or response.status_code in {429, 500, 502, 504}:
                    msg = f"[{attempt}/{max_retries}] Server error {response.status_code} for {url}. Retrying in {retry_interval:.1f}s."
                else:
                    msg = f"[{attempt}/{max_retries}] Unexpected status {response.status_code} for {url}. Retrying in {retry_interval:.1f}s."

                self.logger.log(msg)
                self.logger.log(f"Params: {params}")
                time.sleep(retry_interval)

            except requests.exceptions.Timeout:
                msg = f"[{attempt}/{max_retries}] Timeout for {url}. Retrying in {retry_interval:.1f}s."
                self.logger.log(msg)
                self.logger.log(f"Params: {params}")
                time.sleep(retry_interval)

            except requests.exceptions.RequestException as e:
                msg = f"[{attempt}/{max_retries}] Request failed: {e}. Retrying in {retry_interval:.1f}s."
                self.logger.log(msg)
                self.logger.log(f"Params: {params}")
                time.sleep(retry_interval)

            # Exponential backoff with jitter
            retry_interval *= random.uniform(1.2, 1.8)

        # If we exhausted retries:
        msg = f"[ERROR] Failed after {max_retries} attempts for {url} with params {params}."
        self.logger.log(msg)
        return {'error': f"Failed after {max_retries} attempts"}

    def _get(self, url: str, params: dict[str], headers: dict[str]) -> dict[str]:
        return self._get_retry(url=url, params=params, headers=headers)
