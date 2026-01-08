import json
import os
from typing import Tuple

import pandas as pd

from src.db.db_manager import DBManager
from src.logging.logger import Logger
from src.scrapers.nba.utils.formatting import fmt_player_data, fmt_stats_data, fmt_game_data, fmt_team_data, \
    clean_tables
from src.scrapers.nba.utils.validation import is_bad_game, is_bad_stat
from src.types.game_types import StatType
from src.types.nba_types import RawGameStats, GameMeta, GameStats, PlayerMeta, PlayerStats, TeamMeta, RawGameMeta, \
    RawTeamData, RawPlayerData
from src.utils.file_io import get_dirs


def parse_boxscore(score: dict[str]) -> RawGameStats:
    metric = list(score.keys())[1]
    return score[metric]


def parse_games(games: dict[str]) -> dict[str]:
    fmt_games = {}
    for card in games['modules'][0]['cards']:
        data = card['cardData']
        game_id = data['gameId']

        fmt_games[game_id] = {
            'meta': {
                'season_yr': data['seasonYear'],
                'season_type': data['seasonType'],
                'game_time': data['gameTimeEastern']
            },
            'home': data['homeTeam'],
            'away': data['awayTeam']
        }

    return fmt_games


def parse_dumped_game_data(game_dir: str, dint: int, game_id: str) \
        -> Tuple[GameMeta, list[GameStats], list[PlayerMeta], list[PlayerStats], list[TeamMeta]]:
    stat_files = os.listdir(game_dir)

    game_meta: GameMeta = None
    seen_players = set()
    player_stats_dict = {}
    game_stats_dict = {}
    player_data: list[PlayerMeta] = []
    team_data: list[TeamMeta] = []

    for stat_file in stat_files:
        fpath = os.path.join(game_dir, stat_file)
        j: RawGameMeta | RawGameStats = json.load(open(fpath))

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
                team: RawTeamData = j[side]
                players: list[RawPlayerData] = team['players']
                team_id = team['teamId']
                is_home = side == 'homeTeam'

                # game stats
                if 'usage' not in stat_file:
                    for stat_type in list(StatType):
                        if stat_type.value not in team:
                            continue

                        if not team[stat_type.value]:
                            # happens on empty games
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


def parse_dumped_data_by_day(games_folder: str, date: str, logger: Logger) -> Tuple[pd.DataFrame]:
    dbm = DBManager(logger=logger)

    seen_players = set(dbm.get_all_player_ids().tolist())
    seen_teams = set(dbm.get_all_team_ids().tolist())

    master_player_meta = []
    master_player_stats = []
    master_team_meta = []
    master_game_meta = []
    master_game_stats = []

    games = get_dirs(os.path.join(games_folder))
    for game in games:
        path = os.path.join(games_folder, game)
        game_meta, game_stats, player_data, player_stats, team_data = parse_dumped_game_data(path, int(date), game)

        for player in player_data:
            if player['player_id'] not in seen_players:
                master_player_meta.append(player)
                seen_players.add(player['player_id'])

        for team in team_data:
            if team['team_id'] not in seen_teams:
                master_team_meta.append(team)
                seen_teams.add(team['team_id'])

        master_player_stats.extend(player_stats)
        master_game_stats.extend(game_stats)
        master_game_meta.append(game_meta)

    game_meta_table, game_stats_table, team_meta_table, player_meta_table, player_stats_table = clean_tables(
        master_game_meta, master_game_stats, master_team_meta, master_player_meta, master_player_stats
    )

    return game_meta_table, team_meta_table, player_meta_table, player_stats_table, game_stats_table
