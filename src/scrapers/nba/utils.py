import json
import os
from datetime import datetime, timedelta
from typing import Any, Tuple

import numpy as np
import pandas as pd

from src.db.constants import SCHEMAS
from src.db.schema import GAMES_META_SCHEMA, TEAMS_SCHEMA, PLAYERS_SCHEMA, GAME_STATS_SCHEMA, PLAYER_STATS_SCHEMA
from src.db.utils import insert_table
from src.logging.logger import Logger
from src.scrapers.nba.constants import N_STAT_TYPES, SEASON_TYPE_MAP, PLAYER_DUPE_COLS, GAME_DUPE_COLS
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.types.game_types import StatType
from src.types.nba_types import RawPlayerData, PlayerStats, RawGameMeta, GameMeta, RawTeamData, \
    TeamMeta, RawGameStats, GameStats, PlayerMeta


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


def generate_dates(start_year:int, start_month:int=1, start_day:int=1) -> list[datetime.date]:
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime.today()

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def date_to_dint(date:datetime.date) -> int:
    return int(date.strftime('%Y%m%d'))


def date_to_lookup(date:datetime.date, date_format="%m/%d/%Y") -> str:
    return date.strftime(date_format)


def get_dirs(dir:str) -> list[str]:
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def get_files(dir:str) -> list[str]:
    return [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]


def is_date_data_complete(dir:str, dint:int) -> bool:
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


def is_game_data_complete(dir:str) -> bool:
    files = get_files(dir)
    return len(files) == N_STAT_TYPES


def stat_type_exists(fpath:str) -> bool:
    return os.path.isfile(fpath)


def fetch_and_save_boxscore(game_id:str,
                            boxscore:str,
                            api:NBAStatsApi,
                            data_path:str,
                            logger:Logger):
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

    fmt_game = parse_boxscore(score=score)
    json.dump(fmt_game, open(stat_type_fpath, 'w'))

    msg = f"[SUCCESS] {game_id}/{boxscore}"
    logger.log(msg)


def fmt_player_data(player:RawPlayerData) -> PlayerMeta:
    return {
        'player_id': player['personId'],
        'player_name': player['firstName'] + ' ' + player['familyName'],
        'player_slug': player['playerSlug']
    }


def fmt_stats_data(stats:dict[str, Any],
                   game_id:str,
                   player_id:int,
                   team_id:int,
                   position:str) -> PlayerStats:
    return {
        'player_id': player_id,
        'team_id': team_id,
        'game_id': game_id,
        'position': position,
        **stats
    }


def fmt_game_data(game: RawGameMeta, dint:int, game_id: str) -> GameMeta:
    return {
        'game_id': game_id,
        'season': game['meta']['season_yr'],
        'season_type': game['meta']['season_type'],
        'season_type_code': SEASON_TYPE_MAP[game['meta']['season_type']].value,
        'dint': dint,
        'date': datetime.strptime(str(dint), '%Y%m%d'),
    }


def fmt_team_data(team: RawTeamData) -> TeamMeta:
    return {
        'team_id': team['teamId'],
        'team_name': team['teamName'],
        'team_slug': team['teamTricode']
    }


def is_bad_game(game: RawGameMeta) -> bool:
    return not ('home' in game and game['home'] and 'away' in game and game['away'])


def is_bad_stat(game: RawGameStats) -> bool:
    for side in ['homeTeam', 'awayTeam']:
        if side not in game:
            return True

        team = game[side]
        if StatType.TOTAL.value not in team:
            return True

        if not team[StatType.TOTAL.value]:
            return True

    return False


def time_to_minutes(time_string:str) -> float:
    x = list(map(int, time_string.split(':')))
    if len(x) == 2:
        total_seconds = x[0] * 60 + x[1]
    else:
        total_seconds = abs(x[0]) * 60

    return total_seconds / 60


def parse_dumped_game_data(game_dir:str, dint:int, game_id:str)\
        -> Tuple[GameMeta, list[GameStats], list[PlayerMeta], list[PlayerStats], list[TeamMeta]]:

    stat_files = os.listdir(game_dir)

    game_meta:GameMeta = None
    seen_players = set()
    player_stats_dict = {}
    game_stats_dict = {}
    player_data:list[PlayerMeta] = []
    team_data:list[TeamMeta] = []

    for stat_file in stat_files:
        fpath = os.path.join(game_dir, stat_file)
        j:RawGameMeta|RawGameStats = json.load(open(fpath))

        if 'meta' in stat_file:
            if is_bad_game(j):
                continue

            game_meta = fmt_game_data(j, dint, game_id)

            for side in ['home', 'away']:
                team = j[side]
                team_data.append(fmt_team_data(team))

        else:
            if is_bad_stat(j):
                continue

            for side in ['homeTeam', 'awayTeam']:
                team:RawTeamData = j[side]
                players:list[RawPlayerData] = team['players']
                team_id = team['teamId']
                is_home = side == 'homeTeam'

                # game stats
                if 'usage' not in stat_file:
                    for stat_type in list(StatType):
                        if stat_type.value not in team:
                            continue

                        key = str(game_id) + '_' + str(team_id) + '_' + stat_type
                        if key in game_stats_dict:
                            game_stats_dict[key] = team[stat_type.value] | game_stats_dict[key]
                        else:
                            game_stats_dict[key] = team[stat_type.value]
                            game_stats_dict[key]['game_id'] = game_id
                            game_stats_dict[key]['team_id'] = team_id
                            game_stats_dict[key]['is_home'] = is_home
                            game_stats_dict[key]['stat_type'] = stat_type.value

                # player stats
                for player in players:
                    pdata = fmt_player_data(player)

                    if StatType.TOTAL.value not in player:
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


def clean_tables(game_meta, game_stats, team_meta, player_meta, player_stats):
    game_meta_table = pd.DataFrame(game_meta).replace('', np.nan)
    game_stats_table = pd.DataFrame(game_stats).replace('', np.nan)
    team_meta_table = pd.DataFrame(team_meta).replace('', np.nan)
    player_meta_table = pd.DataFrame(player_meta).replace('', np.nan)
    player_stats_table = pd.DataFrame(player_stats).replace('', np.nan)

    game_stats_table['minutes'] = game_stats_table.minutes.replace(np.nan, '00:00')
    player_stats_table['minutes'] = player_stats_table.minutes.replace(np.nan, '00:00')

    game_stats_table['minutes'] = game_stats_table.minutes.apply(time_to_minutes)
    player_stats_table['minutes'] = player_stats_table.minutes.apply(time_to_minutes)

    game_stats_table = game_stats_table.drop(columns=GAME_DUPE_COLS, errors='ignore')
    player_stats_table = player_stats_table.drop(columns=PLAYER_DUPE_COLS, errors='ignore')

    game_stats_table['stat_type'] = game_stats_table.stat_type.apply(lambda s: s.upper())

    return game_meta_table, game_stats_table, team_meta_table, player_meta_table, player_stats_table


def insert_parsed_data_by_day(games_folder:str, db:str, date:str):
    seen_players = set()
    seen_teams = set()

    master_player_meta = []
    master_player_stats = []
    master_team_meta = []
    master_game_meta = []
    master_game_stats = []

    games = get_dirs(os.path.join(games_folder))
    for game in games:
        path = os.path.join(games_folder, game)
        game_meta, game_stats, player_data, player_stats, team_data = parse_dumped_game_data(path, int(date), game)

        for pdata in player_data:
            if not pdata['player_id'] in seen_players:
                master_player_meta.append(pdata)
                seen_players.add(pdata['player_id'])

        for tdata in team_data:
            if not tdata['team_id'] in seen_teams:
                master_team_meta.append(tdata)
                seen_teams.add(tdata['team_id'])

        master_player_stats.extend(player_stats)
        master_game_stats.extend(game_stats)
        master_game_meta.append(game_meta)

    game_meta_table, game_stats_table, team_meta_table, player_meta_table, player_stats_table = clean_tables(
        master_game_meta, master_game_stats, master_team_meta, master_player_meta, master_player_stats
    )

    tables = [game_meta_table, team_meta_table, player_meta_table, player_stats_table, game_stats_table]
    names = ['games', 'teams', 'players', 'player_stats', 'game_stats']

    for table, schema, name in zip(tables, SCHEMAS, names):
        insert_table(table, schema, name, db, drop=False)