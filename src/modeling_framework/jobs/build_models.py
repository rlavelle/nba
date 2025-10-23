import configparser
import datetime
import os.path
import pickle
import time

import pandas as pd

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.modeling_framework.jobs.utils.formatting import fmt_player_data, fmt_diff_data
from src.modeling_framework.jobs.utils.money_line_model import build_money_line_model
from src.modeling_framework.jobs.utils.prop_model import build_player_prop_model
from src.modeling_framework.jobs.utils.spread_model import build_spread_model
from src.utils.date import date_to_dint

if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True)

    today = date_to_dint(datetime.date.today())

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        model_path = config.get('MODEL_PATH', 'path')
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        train_game_data = fmt_diff_data(logger=logger, cache=True)
    except Exception as e:
        logger.log(f'[ERROR GENERATING GAME DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        train_player_data = fmt_player_data(logger=logger, cache=True)
    except Exception as e:
        logger.log(f'[ERROR GENERATING PLAYER DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    train_game_data = train_game_data[train_game_data.date < pd.Timestamp(2030, 1, 1)]
    train_player_data = train_player_data[train_player_data.date < pd.Timestamp(2030, 1, 1)]

    prop_model = None
    spread_model = None
    standardizer = None
    money_line_model = None

    try:
        prop_model = build_player_prop_model(train_player_data)
        t = time.time()
        logger.log(f'[BUILT PROP MODEL]: {round((t - start), 2)}s')
    except Exception as e:
        logger.log(f'[ERROR BUILDING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    try:
        spread_model, standardizer = build_spread_model(train_game_data)
        t = time.time()
        logger.log(f'[BUILT SPREAD MODEL]: {round((t - start), 2)}s')
    except Exception as e:
        logger.log(f'[ERROR BUILDING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    try:
        money_line_model = build_money_line_model(train_game_data)
        t = time.time()
        logger.log(f'[BUILT MONEYLINE MODEL]: {round((t - start), 2)}s')
    except Exception as e:
        logger.log(f'[ERROR BUILDING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    if prop_model:
        try:
            path = os.path.join(model_path, 'prop', f'{today}')
            os.makedirs(path, exist_ok=True)
            prop_model.save(fpath=os.path.join(path, f'{prop_model.name}.pkl'))
            logger.log(f'[PROP MODEL SAVED]: {path}')
        except Exception as e:
            logger.log(f'[ERROR SAVING PROP MODEL]: {e}')
            insert_error({'msg': str(e)})

    if spread_model:
        try:
            path = os.path.join(model_path, 'spread', f'{today}')
            os.makedirs(path, exist_ok=True)
            spread_model.save(fpath=os.path.join(path, f'{spread_model.name}.pkl'))
            pickle.dump(standardizer, open(os.path.join(path, 'std.pkl'), 'wb'))
            logger.log(f'[SPREAD MODEL SAVED]: {path}')
        except Exception as e:
            logger.log(f'[ERROR SAVING SPREAD MODEL]: {e}')
            insert_error({'msg': str(e)})

    if money_line_model:
        try:
            path = os.path.join(model_path, 'moneyline', f'{today}')
            os.makedirs(path, exist_ok=True)
            money_line_model.save(fpath=os.path.join(path, f'{money_line_model.name}.pkl'))
            logger.log(f'[MONEYLINE MODEL SAVED]: {path}')
        except Exception as e:
            logger.log(f'[ERROR SAVING MONEYLINE MODEL]: {e}')
            insert_error({'msg': str(e)})

    end = time.time()
    logger.log(f'[MODELS BUILT AND SAVED]: {round((end - start), 2)}s')
