import configparser
import datetime
import os
import pickle
import time

import pandas as pd

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.feature_engineering.utils.build_features import build_game_lvl_fts, build_player_lvl_fts
from src.logging.logger import Logger
from src.modeling_framework.jobs.utils.formatting import fmt_diff_data, fmt_player_data
from src.modeling_framework.jobs.utils.prop_model import predict_player_prop_model
from src.utils.date import date_to_dint

# TODO: need to experiment with pre season data... will have to go back and scrape

def pretty_print_results():
    pass

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
        ft_data = build_game_lvl_fts(logger=logger, cache=True)
        game_data = fmt_diff_data(ft_data)
    except Exception as e:
        logger.log(f'[ERROR GENERATING GAME DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        ft_data = build_player_lvl_fts(logger=logger, cache=True)
        player_data = fmt_player_data(ft_data)
    except Exception as e:
        logger.log(f'[ERROR GENERATING PLAYER DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    test_game_data = game_data[game_data.date == pd.Timestamp(2030, 1, 1)]
    test_player_data = player_data[player_data.date == pd.Timestamp(2030, 1, 1)]

    train_game_data = game_data[game_data.date < pd.Timestamp(2030, 1, 1)]
    train_player_data = player_data[player_data.date < pd.Timestamp(2030, 1, 1)]

    prop_model = None
    spread_model = None
    standardizer = None
    money_line_model = None

    try:
        path = os.path.join(model_path, 'prop', f'{today}')
        prop_model = pickle.load(open(os.path.join(path, f'player_ppm_diff_model.pkl'), 'rb'))
    except Exception as e:
        logger.log(f'[ERROR LOADING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    try:
        path = os.path.join(model_path, 'spread', f'{today}')
        spread_model = pickle.load(open(os.path.join(path, f'xgb_spread_model.pkl'), 'rb'))
        standardizer = pickle.load(open(os.path.join(path, 'std.pkl'), 'rb'))
    except Exception as e:
        logger.log(f'[ERROR LOADING SPREAD MODEL]: {e}')
        insert_error({'msg': str(e)})

    try:
        path = os.path.join(model_path, 'moneyline', f'{today}')
        money_line_model = pickle.load(open(os.path.join(path, f'xgb_ml_model.pkl'), 'rb'))
    except Exception as e:
            logger.log(f'[ERROR LOADING MONEYLINE MODEL]: {e}')
            insert_error({'msg': str(e)})

    if prop_model:
        prop_preds = predict_player_prop_model(prop_model, train_player_data, test_player_data)

    if spread_model and standardizer:
        # TODO: need to find combos of teams (and home and away) for the preds
        pass

    if money_line_model:
        # TODO: same as spreads need to find team_id combos and H/A
        pass

    logger.log(f'{pretty_print_results()}')

    end = time.time()
    logger.log(f'[PREDICTION COMPLETE]: {round((end - start), 2)}s')


