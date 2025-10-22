from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.scrapers.nba.constants import SEASON_TYPE_MAP, GAME_DUPE_COLS, PLAYER_DUPE_COLS
from src.utils.date import time_to_minutes
from src.types.nba_types import RawPlayerData, PlayerMeta, PlayerStats, RawGameMeta, GameMeta, RawTeamData, TeamMeta


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
