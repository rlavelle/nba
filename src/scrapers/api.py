import configparser
import random
import time
from typing import Any

import requests

from src.config import CONFIG_PATH
from src.logging.logger import Logger


class API:
    def __init__(self, logger):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)

        self.logger = logger

        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

    def _get_retry(self,
                   url: str,
                   params: dict[str,str],
                   headers: dict[str]|None,
                   max_retries=10,
                   retry_interval=10,
                   timeout=10) -> dict[str]:

        for attempt in range(1, max_retries + 1):
            try:
                if headers:
                    response = requests.get(url, params=params, headers=headers, timeout=timeout)
                else:
                    response = requests.get(url, params=params, timeout=timeout)

                if response.status_code == 200:
                    return response.json()

                # Handle known retryable status codes
                if response.status_code in {503, 429, 500, 502, 504}:
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

    def _get(self, url: str, params: dict[str,str], headers: dict[str]|None) -> dict[str]|Any:
        return self._get_retry(url=url, params=params, headers=headers)
