import configparser
import datetime
import hashlib
import json
import os
import random
import time
from typing import Any, Union

import requests

from src.config import CONFIG_PATH
from src.logging.logger import Logger


class API:
    def __init__(self, logger):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)
        self.cache_path = self.config.get('DATA_PATH', 'cache_folder')
        self.logger = logger

        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

    @staticmethod
    def _hash_dict(d: dict) -> str:
        """Create a short hash from a dict so it can be safely used in a filename."""
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _hash_url(url: str) -> str:
        """Create a short hash from a URL to safely use in a filename."""
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def _today_iso() -> str:
        return datetime.date.today().isoformat()  # YYYY-MM-DD

    def _get_retry(self,
                   url: str,
                   params: dict[str, str],
                   headers: Union[dict[str], None],
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

    def _get(self, url: str, params: dict[str, str], headers: Union[dict[str], None], cache: bool = False) -> Union[
        dict[str], Any]:
        if not cache:
            return self._get_retry(url=url, params=params, headers=headers)

        os.makedirs(self.cache_path, exist_ok=True)

        param_hash = API._hash_dict(params) if params else "noparams"
        header_hash = API._hash_dict(headers) if headers else "noheaders"
        url_hash = API._hash_url(url) if url else "nourl"
        filename = f"{API._today_iso()}_{url_hash}_{param_hash}_{header_hash}.json"
        cache_file = os.path.join(self.cache_path, filename)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.logger.log(f'[SUCCESS ON CACHE HIT]: {cache_file}')
                    return json.load(f)
            except Exception:
                pass

        response = self._get_retry(url=url, params=params, headers=headers)
        try:
            with open(cache_file, "w") as f:
                json.dump(response, f)
        except Exception as e:
            self.logger.log(f'[ERROR ON SAVING CACHE]: {e}')

        return response
