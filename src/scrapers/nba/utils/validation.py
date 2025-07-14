import json
import os

from src.scrapers.nba.constants import N_STAT_TYPES
from src.scrapers.nba.utils.file_io import get_dirs, get_files
from src.types.game_types import StatType
from src.types.nba_types import RawGameMeta, RawGameStats


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
    if not os.path.exists(dir):
        return False

    files = get_files(dir)
    return len(files) == N_STAT_TYPES


def stat_type_exists(fpath:str) -> bool:
    return os.path.isfile(fpath)


def is_empty_game(game: RawGameMeta) -> bool:
    return game['home']['score'] == 0 and game['away']['score'] == 0


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
