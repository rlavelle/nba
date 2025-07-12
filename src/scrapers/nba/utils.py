import json
import os
from datetime import datetime, timedelta

import pandas as pd

N_STAT_TYPES = 7 # number of types of stats pulled per game

SEASON_TYPE_MAP = {
    'Regular Season': '00',
    'Playoffs': '01',
    'All-Star': '02',
    'Preseason': '03',
    'Summer League': '04',
    'PlayIn': '05',
    'IST Championship': '06'
}

PLAYER_DUPE_COLS = [
    "percentagePoints",
    "percentageFieldGoalsMade",
    "percentageFieldGoalsAttempted",
    "percentageThreePointersMade",
    "percentageThreePointersAttempted",
    "percentageFreeThrowsMade",
    "percentageFreeThrowsAttempted",
    "percentageReboundsOffensive",
    "percentageReboundsDefensive",
    "percentageReboundsTotal",
    "percentageAssists",
    "percentageTurnovers",
    "percentageSteals",
    "percentageBlocks",
    "percentageBlocksAllowed"
]

GAME_DUPE_COLS = [
    "estimatedTeamTurnoverPercentage"  # Same as teamTurnoverPercentage
]

def parse_boxscore(score:dict[str]) -> dict[str]:
    metric = list(score.keys())[1]
    return score[metric]


def parse_games(games:dict[str]) -> dict[str]:
    fmt_games = {}
    for card in games['modules'][0]['cards']:
        data = card['cardData']
        game_id = data['gameId']

        fmt_games[game_id] = {
            'meta':{
                'season_yr':data['seasonYear'],
                'season_type':data['seasonType'],
                'game_time':data['gameTimeEastern']
            },
            'home': data['homeTeam'],
            'away': data['awayTeam']
        }

    return fmt_games


def generate_dates(start_year, start_month=1, start_day=1):
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime.today()

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def date_to_dint(date):
    return int(date.strftime('%Y%m%d'))


def date_to_lookup(date, date_format="%m/%d/%Y"):
    return date.strftime(date_format)


def get_dirs(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def get_files(dir):
    return [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]


def is_date_data_complete(dir, dint):
    if not os.path.isdir(dir):
        return False
    game_file = os.path.join(dir,f'{dint}_games.json')
    if not os.path.isfile(game_file):
        return False

    games = json.load(open(game_file))
    ngames = len(games)
    game_dirs = get_dirs(dir)

    if ngames > len(game_dirs):
        return False

    for game_dir in game_dirs:
        path = os.path.join(dir, game_dir)
        if not is_game_data_complete(path):
            return False

    return True


def is_game_data_complete(dir):
    files = get_files(dir)
    return len(files) == N_STAT_TYPES


def stat_type_exists(fpath):
    return os.path.isfile(fpath)


def fetch_and_save_boxscore(game_id, boxscore, api, util, data_path, logger):
    stat_type_fpath = os.path.join(data_path, f'{game_id}_{boxscore}_stats.json')
    if stat_type_exists(stat_type_fpath):
        return

    score = api.get_boxscore(game_id=game_id, endpoint=boxscore, period='full_game')

    if 'error' in score:
        print(f'bad api hit on {game_id}')
        logger.log(f'bad api hit on {game_id}')
        return

    validate_game = score['meta']['request'].split('/')[4]
    assert game_id == validate_game, f'{game_id} <> {validate_game}'

    fmt_game = util.parse_boxscore(score=score)
    json.dump(fmt_game, open(stat_type_fpath, 'w'))

    msg = f"[SUCCESS] {game_id}/{boxscore}"
    logger.log(msg)


def fmt_player_data(player):
    return {
        'player_id': player['personId'],
        'player_name': player['firstName'] + ' ' + player['familyName'],
        'player_slug': player['playerSlug']
    }


def fmt_stats_data(stats, game_id, player_id, team_id, position):
    return {
        'player_id': player_id,
        'team_id': team_id,
        'game_id': game_id,
        'position': position,
        **stats
    }


def fmt_game_data(game, date, game_id):
    return {
        'game_id': game_id,
        'season': game['meta']['season_yr'],
        'season_type': game['meta']['season_type'],
        'season_type_code': SEASON_TYPE_MAP[game['meta']['season_type']],
        'dint': date,
        'date': datetime.strptime(str(date), '%Y%m%d'),
    }


def fmt_team_data(team):
    return {
        'team_id': team['teamId'],
        'team_name': team['teamName'],
        'team_slug': team['teamTricode']
    }


def is_bad_game(game):
    return not ('home' in game and game['home'] and 'away' in game and game['away'])


def is_bad_stat(game):
    for side in ['homeTeam', 'awayTeam']:
        if side not in game:
            return True

        team = game[side]
        if 'statistics' not in team:
            return True

        if not team['statistics']:
            return True

    return False

def time_to_minutes(time_string):
    x = list(map(int, time_string.split(':')))
    if len(x) == 2:
        total_seconds = x[0] * 60 + x[1]
    else:
        total_seconds = abs(x[0]) * 60

    return total_seconds / 60

def parse_dumped_game_data(game_dir, date, game_id):
    stat_files = os.listdir(game_dir)

    game_meta = None
    seen_players = set()
    player_stats_dict = {}
    game_stats_dict = {}
    player_data = []
    team_data = []

    for stat_file in stat_files:
        fpath = os.path.join(game_dir, stat_file)
        j = json.load(open(fpath))

        if 'meta' in stat_file:
            if is_bad_game(j):
                continue

            game_meta = fmt_game_data(j, date, game_id)

            for side in ['home', 'away']:
                team = j[side]
                team_data.append(fmt_team_data(team))

        else:
            if is_bad_stat(j):
                continue

            for side in ['homeTeam', 'awayTeam']:
                team = j[side]
                players = team['players']
                team_id = team['teamId']
                is_home = side == 'homeTeam'

                # game stats
                if 'usage' not in stat_file:
                    for stat_type in ['statistics', 'starters', 'bench']:
                        if stat_type not in team:
                            continue

                        key = str(game_id) + '_' + str(team_id) + '_' + stat_type
                        if key in game_stats_dict:
                            game_stats_dict[key] = team[stat_type] | game_stats_dict[key]
                        else:
                            game_stats_dict[key] = team[stat_type]
                            game_stats_dict[key]['game_id'] = game_id
                            game_stats_dict[key]['team_id'] = team_id
                            game_stats_dict[key]['is_home'] = is_home
                            game_stats_dict[key]['stat_type'] = stat_type

                # player stats
                for player in players:
                    pdata = fmt_player_data(player)

                    if 'statistics' not in player:
                        continue

                    pid = pdata['player_id']
                    if pid not in seen_players:
                        seen_players.add(pid)
                        player_data.append(pdata)

                    player_stat_data = fmt_stats_data(player['statistics'], game_id, pid, team_id, player['position'])

                    if pid in player_stats_dict:
                        player_stats_dict[pid] = player_stat_data | player_stats_dict[pid]
                    else:
                        player_stats_dict[pid] = player_stat_data

    player_stats = list(player_stats_dict.values())
    game_stats = list(game_stats_dict.values())

    return game_meta, game_stats, player_data, player_stats, team_data
