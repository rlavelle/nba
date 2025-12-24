import configparser
import datetime
import os
import pickle

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.utils.date import date_to_dint


def get_cache_dir():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config.get('DATA_PATH', 'cache_folder')


def gen_cache_file(f, date:int=None):
    curr_date = date if date else date_to_dint(datetime.date.today())
    return os.path.join(get_cache_dir(), f'{curr_date}_{f}.pkl')


def check_cache(f:str, logger:Logger=None, date:int=None):
    cache_file = gen_cache_file(f, date=date)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as file:
                logger.log(f'[SUCCESS ON CACHE HIT]: {cache_file}')
                return pickle.load(file)
        except Exception as e:
            logger.log(f'[ERROR LOADING CACHE]: {e}')
            insert_error({'msg': str(e)})
            return None
