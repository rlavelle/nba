#!/usr/bin/env python3
import argparse
import configparser
import datetime
import os
import pickle
import time

import pandas as pd

from src.config import CONFIG_PATH
from src.db.db_manager import DBManager
from src.db.utils import insert_error
from src.feature_engineering.utils.build_features import build_game_lvl_fts, build_player_lvl_fts
from src.logging.logger import Logger
from src.modeling_framework.jobs.utils.formatting import fmt_diff_data, fmt_player_data, \
    pretty_print_results, send_results
from src.modeling_framework.jobs.utils.money_line_model import insert_ml_results, \
    build_money_line_results
from src.modeling_framework.jobs.utils.prop_model import insert_prop_results, \
    build_player_prop_results
from src.utils.date import date_to_dint

# TODO: need to experiment with pre season data... will have to go back and scrape


if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True, admin=True)
    logger.log(f'[STARTING PREDICTIONS]')

    parser = argparse.ArgumentParser(description='NBA prediction script')
    parser.add_argument('--date', type=str, help='Date to pull in YYYY-MM-DD format (default: today)')
    parser.add_argument('--skip-save', action='store_true', help='Skip result pkl saving')
    parser.add_argument('--skip-insert', action='store_true', help='Skip result table insertion')
    parser.add_argument('--offline', action='store_true', help='offline testing')
    parser.add_argument('--admin', action='store_true', help='admin only email')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.date.today()
    curr_date = date_to_dint(date)
    nxt_date = date_to_dint(date + datetime.timedelta(days=1))

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        model_path = config.get('MODEL_PATH', 'path')
    except Exception as e:
        logger.log(f'[CONFIG LOAD ERROR]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        game_data = build_game_lvl_fts(logger=logger, cache=True, date=curr_date)
    except Exception as e:
        logger.log(f'[ERROR GENERATING GAME DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    try:
        ft_data = build_player_lvl_fts(logger=logger, cache=True, date=curr_date)
        player_data = fmt_player_data(ft_data)
    except Exception as e:
        logger.log(f'[ERROR GENERATING PLAYER DATA]: {e}')
        insert_error({'msg': str(e)})
        exit()

    test_game_data = game_data[game_data.date == pd.Timestamp(2030, 1, 1)]
    test_player_data = player_data[player_data.date == pd.Timestamp(2030, 1, 1)]

    train_game_data = game_data[game_data.date < pd.Timestamp(2030, 1, 1)]
    train_game_data = fmt_diff_data(train_game_data)

    train_player_data = player_data[player_data.date < pd.Timestamp(2030, 1, 1)]

    prop_model = None
    # spread_model = None
    # standardizer = None
    money_line_model = None

    try:
        path = os.path.join(model_path, 'prop', f'{curr_date}')
        prop_model = pickle.load(open(os.path.join(path, f'player_ppm_diff_model.pkl'), 'rb'))
    except Exception as e:
        logger.log(f'[ERROR LOADING PROP MODEL]: {e}')
        insert_error({'msg': str(e)})

    # try:
    #     path = os.path.join(model_path, 'spread', f'{curr_date}')
    #     spread_model = pickle.load(open(os.path.join(path, f'xgb_spread_model.pkl'), 'rb'))
    #     standardizer = pickle.load(open(os.path.join(path, 'std.pkl'), 'rb'))
    # except Exception as e:
    #     logger.log(f'[ERROR LOADING SPREAD MODEL]: {e}')
    #     insert_error({'msg': str(e)})

    try:
        path = os.path.join(model_path, 'moneyline', f'{curr_date}')
        money_line_model = pickle.load(open(os.path.join(path, f'xgb_ml_model.pkl'), 'rb'))
    except Exception as e:
            logger.log(f'[ERROR LOADING MONEYLINE MODEL]: {e}')
            insert_error({'msg': str(e)})

    dbm = DBManager(logger=logger)
    # spread_odds = None
    money_line_odds = None
    prop_odds = None

    try:
        prop_odds = dbm.get_prop_odds()
    except Exception as e:
        logger.log(f'[ERROR LOADING PROP ODDS]: {e}')
        insert_error({'msg': str(e)})

    # try:
    #     spread_odds = dbm.get_spread_odds()
    # except Exception as e:
    #     logger.log(f'[ERROR LOADING SPREAD ODDS]: {e}')
    #     insert_error({'msg': str(e)})

    try:
        money_line_odds = dbm.get_money_line_odds()
    except Exception as e:
        logger.log(f'[ERROR LOADING MONEYLINE ODDS]: {e}')
        insert_error({'msg': str(e)})

    prop_results = build_player_prop_results(
        prop_model, prop_odds, nxt_date, train_player_data, test_player_data, logger
    )

    # TODO: rebuild spread model
    #spread_results = build_spread_results(...)

    # TODO: patched the patched
    ml_results = build_money_line_results(
        money_line_model, money_line_odds, nxt_date, test_game_data, logger
    )

    msg_md, msg_html = pretty_print_results(prop_results, ml_results)
    logger.log(msg_md)

    if not args.offline:
        send_results(f'NBA Results {datetime.date.today()}', msg_html, args.admin)

    if not args.skip_save:
        path = os.path.join(model_path, 'prop', f'{curr_date}')
        prop_results.to_pickle(os.path.join(path, 'prop_preds.pkl'))

        # path = os.path.join(model_path, 'spread', f'{curr_date}')
        # spread_results.to_pickle(os.path.join(path, 'spread_preds.pkl'))

        path = os.path.join(model_path, 'moneyline', f'{curr_date}')
        ml_results.to_pickle(os.path.join(path, 'ml_preds.pkl'))

    if not args.skip_insert:
        try:
            ml_results['dint'] = curr_date
            insert_ml_results(ml_results)
        except Exception as e:
            logger.log(f'[ERROR INSERTING ML RESULTS]: {e}')
            insert_error({'msg': str(e)})

        try:
            prop_results['dint'] = curr_date
            prop_results = prop_results[~prop_results.price.isna()]
            insert_prop_results(prop_results)
        except Exception as e:
            logger.log(f'[ERROR INSERTING PROP RESULTS]: {e}')
            insert_error({'msg': str(e)})

    end = time.time()
    logger.log(f'[PREDICTION COMPLETE]: {round((end - start), 2)}s')

    if not args.offline:
        logger.email_log()
