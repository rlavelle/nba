#!/usr/bin/env python3
import argparse
import configparser
import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.config import CONFIG_PATH
from src.db.db_manager import DBManager
from src.db.utils import insert_error
from src.feature_engineering.utils.build_features import build_game_lvl_fts, build_player_lvl_fts
from src.logging.email_sender import EmailSender
from src.logging.logger import Logger
from src.modeling_framework.jobs.utils.formatting import fmt_diff_data, fmt_player_data
from src.modeling_framework.jobs.utils.money_line_model import predict_money_line_model
from src.modeling_framework.jobs.utils.prop_model import predict_player_prop_model
from src.modeling_framework.jobs.utils.spread_model import predict_spread_model
from src.utils.date import date_to_dint

# TODO: need to experiment with pre season data... will have to go back and scrape

def pretty_print_results(prop_r, ml_r):
    dbm = DBManager()
    players = dbm.get_players()
    teams = dbm.get_teams()

    if prop_r is not None:
        prop_r = prop_r[~prop_r.price.isna()]
        prop_r['delta_pct'] = (prop_r.preds / prop_r.point) - 1
        prop_r = prop_r[['player_id', 'bookmaker',
                         'description', 'price', 'point', 'preds', 'delta_pct']].copy()
        prop_r = pd.merge(prop_r, players, on='player_id', how='left')
        prop_r = prop_r.drop(columns=['player_slug'])

    # if spread_r is not None:
    #     spread_r = spread_r[['team_id', 'bookmaker', 'price', 'point', 'preds']].copy()
    #     spread_r = pd.merge(spread_r, teams, on='team_id', how='left')

    if ml_r is not None:
        ml_r['vegas_preds'] = 1 / ml_r.price
        ml_r['delta_pct'] = (ml_r.preds / ml_r.vegas_preds) - 1
        ml_r = ml_r[['team_id', 'bookmaker', 'price', 'vegas_preds', 'preds', 'delta_pct', 'is_home']].copy()
        ml_r = pd.merge(ml_r, teams, on='team_id', how='left')

    md = f"""
    # NBA Results {datetime.date.today()}
    
    ### PLAYER PROPS
    {prop_r.to_markdown(index=False)}
    
    ### MONEY LINE
    {ml_r.to_markdown(index=False)}
    
    -------------------------------------
    *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    html = f"""
        # NBA Results {datetime.date.today()}

        ### PLAYER PROPS
        {prop_r.to_html(index=False)}

        ### MONEY LINE
        {ml_r.to_html(index=False)}

        -------------------------------------
        *Report generated automatically, Rowan Lavelle is NOT liable for your losses and gambling addiction.
    """

    if ml_r.empty and prop_r.empty:
        return "No odds available today, go home", "No odds available today, go home"

    return md, html


def prep_odds(odds: pd.DataFrame, bookmakers: list[str], tomorrow:int):
    odds = odds[(odds.bookmaker.isin(bookmakers)) & (odds.dint == tomorrow)]
    odds = odds.drop(columns=['last_update', 'dint'])
    return odds.drop_duplicates(keep='first')


# TODO: this is a bad way to reconcile games.... lol
def format_testing_data(odds, test_data):
    tmp = test_data.copy()

    odds['is_home'] = (odds.index % 2 == 0).astype(int)
    odds['game_id'] = odds.index // 2

    tmp = tmp.drop(columns=['is_home', 'game_id'])

    odds['team_id'] = odds.team_id.astype(int)
    tmp['team_id'] = tmp.team_id.astype(int)

    tmp_home = pd.merge(odds[['is_home', 'game_id', 'team_id']], tmp,  on=['team_id'],  how='left')

    odds['is_home'] = np.where(odds.is_home==1, 0, 1)
    tmp_away = pd.merge(odds[['is_home', 'game_id', 'team_id']], tmp,  on=['team_id'],  how='left')

    return fmt_diff_data(tmp_home), fmt_diff_data(tmp_away)


def send_results(msg):
    email_sender = EmailSender()
    email_sender.read_recipients_from_file()
    email_sender.set_subject(f'NBA Results {datetime.date.today()}')
    email_sender.set_body(msg)
    email_sender.send_email(admin=False)


if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True, admin=True)
    logger.log(f'[STARTING PREDICTIONS]')

    parser = argparse.ArgumentParser(description='NBA prediction script')
    parser.add_argument('--date', type=str, help='Date to pull in YYYY-MM-DD format (default: today)')
    parser.add_argument('--skip-save', action='store_true', help='Skip model saving')
    parser.add_argument('--offline', action='store_true', help='offline testing')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.date.today()
    curr_date = date_to_dint(date)
    curr_nxt_date = date_to_dint(date + datetime.timedelta(days=1))

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

    prop_results = None
    if prop_model is not None and prop_odds is not None:
        prop_preds = None
        try:
            prop_preds = predict_player_prop_model(prop_model, train_player_data, test_player_data)
            prop_preds = pd.DataFrame(prop_preds, columns=['preds'])
            prop_preds['player_id'] = test_player_data.player_id

        except Exception as e:
            logger.log(f'[ERROR PREDICTING PLAYER PROPS]: {e}')
            insert_error({'msg': str(e)})

        if prop_preds is not None:
            # TODO: build this out to all?
            prop_odds = prep_odds(prop_odds, bookmakers=['draftkings'], tomorrow=curr_nxt_date)
            prop_results = pd.merge(prop_preds, prop_odds, on='player_id', how='left')
            logger.log(f'[MISSING PLAYERS FROM MODEL]: {prop_odds.price.isna().sum()}')

    # TODO: rebuild spread model
    # spread_results = None
    # if spread_model is not None and standardizer is not None and spread_odds is not None:
    #     spread_odds = prep_odds(spread_odds, bookmakers=['draftkings'], tomorrow=curr_nxt_date)
    #     test_spread_data,_ = format_testing_data(spread_odds, test_game_data)
    #
    #     spread_preds = None
    #     try:
    #         spread_preds = predict_spread_model(spread_model,
    #                                             standardizer,
    #                                             train_game_data,
    #                                             test_spread_data)
    #         spread_preds = pd.DataFrame(spread_preds, columns=['preds'])
    #         #spread_preds.columns = ['preds']
    #         spread_preds['team_id'] = test_spread_data.team_id
    #
    #     except Exception as e:
    #         logger.log(f'[ERROR PREDICTING SPREADS]: {e}')
    #         insert_error({'msg': str(e)})
    #
    #     if spread_preds is not None:
    #         spread_results = pd.merge(spread_preds, spread_odds, on='team_id', how='left')

    ml_results = None
    if money_line_model is not None and money_line_odds is not None:
        money_line_odds = prep_odds(money_line_odds, bookmakers=['draftkings'], tomorrow=curr_nxt_date)
        test_ml_data_home, test_ml_data_away = format_testing_data(money_line_odds, test_game_data)

        ml_preds_home = None
        ml_preds_away = None
        try:
            # NOTE: in theory, the ML model is trained on diff to predict win / loss
            #       so if its generalized well it should be able to predict the flip...
            ml_preds_home = predict_money_line_model(money_line_model,
                                                     test_ml_data_home)
            ml_preds_away = predict_money_line_model(money_line_model,
                                                     test_ml_data_away)

            ml_preds_home = pd.DataFrame(ml_preds_home, columns=['preds'])
            ml_preds_away = pd.DataFrame(ml_preds_away, columns=['preds'])

            ml_preds_home['team_id'] = test_ml_data_home.team_id
            ml_preds_away['team_id'] = test_ml_data_away.team_id

        except Exception as e:
            logger.log(f'[ERROR PREDICTING MONEYLINE]: {e}')
            insert_error({'msg': str(e)})

        if ml_preds_home is not None and ml_preds_away is not None:
            ml_results1 = pd.merge(ml_preds_home, money_line_odds, on='team_id', how='left')
            ml_results2 = pd.merge(ml_preds_away, money_line_odds, on='team_id', how='left')
            ml_results = pd.concat([ml_results1, ml_results2]).sort_values(by=['game_id', 'is_home'])

    msg_md, msg_html = pretty_print_results(prop_results, ml_results)
    logger.log(msg_md)

    if not args.offline:
        send_results(msg_html)

    if not args.skip_save:
        path = os.path.join(model_path, 'prop', f'{curr_date}')
        prop_results.to_pickle(os.path.join(path, 'prop_preds.pkl'))

        # path = os.path.join(model_path, 'spread', f'{curr_date}')
        # spread_results.to_pickle(os.path.join(path, 'spread_preds.pkl'))

        path = os.path.join(model_path, 'moneyline', f'{curr_date}')
        ml_results.to_pickle(os.path.join(path, 'ml_preds.pkl'))

    end = time.time()
    logger.log(f'[PREDICTION COMPLETE]: {round((end - start), 2)}s')

    if not args.offline:
        logger.email_log()
