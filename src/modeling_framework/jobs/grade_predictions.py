import argparse
import datetime
import time

from src.db.db_manager import DBManager
from src.db.utils import insert_error
from src.logging.logger import Logger
from src.modeling_framework.framework.dataloader import NBADataLoader
from src.modeling_framework.jobs.utils.formatting import pretty_print_grading, send_results
from src.modeling_framework.jobs.utils.grading import format_player_data, format_game_data
from src.types.player_types import PlayerType
from src.utils.date import date_to_dint

if __name__ == "__main__":
    start = time.time()
    logger = Logger(fpath='cron_path', daily_cron=True, admin=True)
    logger.log(f'[STARTING GRADING]')

    parser = argparse.ArgumentParser(description='NBA prediction script')
    parser.add_argument('--date', type=str, help='Date to pull in YYYY-MM-DD format (default: today)')
    parser.add_argument('--offline', action='store_true', help='offline testing')
    parser.add_argument('--admin', action='store_true', help='admin only email')
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.date.today()
    curr_date = date_to_dint(date)
    prev_date = date_to_dint(date + datetime.timedelta(days=-1))

    dbm = DBManager(logger=logger)

    try:
        prop_results = dbm.get_prop_results()
    except Exception as e:
        logger.log(f'[ERROR LOADING PROP RESULTS]: {e}')
        insert_error({'msg': str(e)})
        logger.email_log()
        exit()

    try:
        ml_results = dbm.get_money_line_results()
    except Exception as e:
        logger.log(f'[ERROR LOADING MONEYLINE RESULTS]: {e}')
        insert_error({'msg': str(e)})
        logger.email_log()
        exit()

    try:
        data_loader = NBADataLoader()
        data_loader.load_data()

        games = data_loader.get_data('games')
        game_data = format_game_data(games, ml_results)

        ptypes = (PlayerType.STARTER, PlayerType.STARTER_PLUS, PlayerType.PRIMARY_BENCH)
        player_data = data_loader.get_player_type(ptypes=ptypes)
        player_data = format_player_data(player_data, prop_results)
    except Exception as e:
        logger.log(f'[ERROR LOADING FROM NBA DATA LOADER]: {e}')
        insert_error({'msg': str(e)})
        logger.email_log()
        exit()

    # total results
    game_wins = game_data[game_data.win == 1].copy()
    player_wins = player_data[player_data.win == 1].copy()

    # prev day results
    game_wins_prev = game_wins[game_wins.dint == prev_date].copy()
    player_wins_prev = player_wins[player_wins.dint == prev_date].copy()

    try:
        msg_md, msg_html = pretty_print_grading(
            game_wins, player_wins, game_wins_prev, player_wins_prev
        )
        logger.log(msg_md)

    except Exception as e:
        logger.log(f'[ERROR ON PRETTY PRINT]: {e}')
        insert_error({'msg': str(e)})
        logger.email_log()
        exit()

    if not args.offline:
        send_results(f'NBA Bet Grading {datetime.date.today()}', msg_html, args.admin)

    if not args.offline:
        logger.email_log()
