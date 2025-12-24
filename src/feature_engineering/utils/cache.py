import configparser
import datetime
import os
import pickle

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.utils.date import date_to_dint

# TODO: all cache functions should accept dates so that we can cache load
#       historical data for offline testing

def get_cache_dir():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config.get('DATA_PATH', 'cache_folder')


def gen_cache_file(f):
    today = date_to_dint(datetime.date.today())
    return os.path.join(get_cache_dir(), f'{today}_{f}.pkl')


def check_cache(f, logger=None):
    cache_file = gen_cache_file(f)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as file:
                logger.log(f'[SUCCESS ON CACHE HIT]: {cache_file}')
                return pickle.load(file)
        except Exception as e:
            logger.log(f'[ERROR LOADING CACHE]: {e}')
            insert_error({'msg': str(e)})
            return None
