import argparse
import configparser
import datetime
import os.path
import pickle
import time

import pandas as pd

from src.config import CONFIG_PATH
from src.db.utils import insert_error
from src.feature_engineering.utils.build_features import build_game_lvl_fts, build_player_lvl_fts
from src.logging.logger import Logger
from src.modeling_framework.jobs.utils.formatting import fmt_player_data, fmt_diff_data
from src.modeling_framework.jobs.utils.money_line_model import build_money_line_model
from src.modeling_framework.jobs.utils.prop_model import build_player_prop_model
from src.modeling_framework.jobs.utils.spread_model import build_spread_model
from src.utils.date import date_to_dint

if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True)

    parser = argparse.ArgumentParser(description='NBA model building script')
    parser.add_argument('--date', type=str, help='Date to pull in YYYY-MM-DD format (default: today)')
    parser.add_argument('--skip-save', action='store_true', help='Skip model saving')
    parser.add_argument('--offline', action='store_true', help='offline testing')
    parser.add_argument('--recent', action='store_true', help='use most recent feature cache file')
    parser.add_argument('--cache', action='store_true', help='use cache')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.date.today()
    curr_date = date_to_dint(date)

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        model_path = config.get('MODEL_PATH', 'path')
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})

        if not args.offline:
            logger.email_log()
        exit()

    try:
        ft_data = build_game_lvl_fts(logger=logger, cache=args.cache, date=curr_date, recent=args.recent)
        game_data = fmt_diff_data(ft_data)
    except Exception as e:
        logger.log(f'[ERROR GENERATING GAME DATA]: {e}')
        insert_error({'msg': str(e)})

        if not args.offline:
            logger.email_log()
        exit()

    try:
        ft_data = build_player_lvl_fts(logger=logger, cache=args.cache, date=curr_date, recent=args.recent)
        player_data = fmt_player_data(ft_data)
    except Exception as e:
        logger.log(f'[ERROR GENERATING PLAYER DATA]: {e}')
        insert_error({'msg': str(e)})

        if not args.offline:
            logger.email_log()
        exit()

    train_game_data = game_data[game_data.date < pd.Timestamp(2030, 1, 1)]
    train_player_data = player_data[player_data.date < pd.Timestamp(2030, 1, 1)]

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

    # TODO: fix spread model
    # try:
    #     spread_model, standardizer = build_spread_model(train_game_data)
    #     t = time.time()
    #     logger.log(f'[BUILT SPREAD MODEL]: {round((t - start), 2)}s')
    # except Exception as e:
    #     logger.log(f'[ERROR BUILDING PROP MODEL]: {e}')
    #     insert_error({'msg': str(e)})

    try:
        money_line_model = build_money_line_model(train_game_data)
        t = time.time()
        logger.log(f'[BUILT MONEYLINE MODEL]: {round((t - start), 2)}s')
    except Exception as e:
        logger.log(f'[ERROR BUILDING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    if prop_model and not args.skip_save:
        try:
            path = os.path.join(model_path, 'prop', f'{curr_date}')
            os.makedirs(path, exist_ok=True)
            prop_model.save(fpath=os.path.join(path, f'{prop_model.name}.pkl'))
            logger.log(f'[PROP MODEL SAVED]: {path}')
        except Exception as e:
            logger.log(f'[ERROR SAVING PROP MODEL]: {e}')
            insert_error({'msg': str(e)})

    # TODO: fix spread model
    # if spread_model and not args.skip_save:
    #     try:
    #         path = os.path.join(model_path, 'spread', f'{curr_date}')
    #         os.makedirs(path, exist_ok=True)
    #         spread_model.save(fpath=os.path.join(path, f'{spread_model.name}.pkl'))
    #         pickle.dump(standardizer, open(os.path.join(path, 'std.pkl'), 'wb'))
    #         logger.log(f'[SPREAD MODEL SAVED]: {path}')
    #     except Exception as e:
    #         logger.log(f'[ERROR SAVING SPREAD MODEL]: {e}')
    #         insert_error({'msg': str(e)})

    if money_line_model and not args.skip_save:
        try:
            path = os.path.join(model_path, 'moneyline', f'{curr_date}')
            os.makedirs(path, exist_ok=True)
            money_line_model.save(fpath=os.path.join(path, f'{money_line_model.name}.pkl'))
            logger.log(f'[MONEYLINE MODEL SAVED]: {path}')
        except Exception as e:
            logger.log(f'[ERROR SAVING MONEYLINE MODEL]: {e}')
            insert_error({'msg': str(e)})

    end = time.time()
    logger.log(f'[MODELS BUILT AND SAVED]: {round((end - start), 2)}s')

    if not args.offline:
        logger.email_log()
