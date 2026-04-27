import json
import os
from enum import Enum

from src.config import CONFIG_PATH
import configparser

from src.scrapers.nba.constants import N_STAT_TYPES
from src.types.game_types import StatType
from src.types.nba_types import RawGameMeta, RawGameStats
from src.utils.file_io import get_dirs, get_files


class DataCompleteness(Enum):
    COMPLETE = "complete"
    NO_DIR = "no_directory"
    NO_META_FILE = "no_meta_file"
    MISSING_GAMES = "missing_games"
    FULL_STATS_MISSED = "full_stats_missed"
    PARTIAL_STATS_MISSED = "partial_stats_missed"


def is_date_data_complete(dir: str, dint: int) -> DataCompleteness:
    if not os.path.isdir(dir):
        return DataCompleteness.NO_DIR

    game_file = os.path.join(dir, f'{dint}_games.json')
    if not os.path.isfile(game_file):
        return DataCompleteness.NO_META_FILE

    games = json.load(open(game_file))
    ngames = len(games)
    game_dirs = get_dirs(dir)

    if ngames > len(game_dirs):
        return DataCompleteness.MISSING_GAMES

    partial_miss = False
    for game_dir in game_dirs:
        path = os.path.join(dir, game_dir)
        result = is_game_data_complete(path)
        if result == DataCompleteness.PARTIAL_STATS_MISSED:
            partial_miss = True
            continue

        if result == DataCompleteness.FULL_STATS_MISSED:
            return result

    if partial_miss:
        return DataCompleteness.PARTIAL_STATS_MISSED

    return DataCompleteness.COMPLETE


def is_game_data_complete(dir: str) -> DataCompleteness:
    if not os.path.exists(dir):
        return DataCompleteness.FULL_STATS_MISSED

    files = get_files(dir)

    if len(files) == 1:
        return DataCompleteness.FULL_STATS_MISSED
    elif len(files) != N_STAT_TYPES:
        return DataCompleteness.PARTIAL_STATS_MISSED

    return DataCompleteness.COMPLETE


def stat_type_exists(fpath: str) -> bool:
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

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    data_path = config.get('DATA_PATH', 'games_folder')

    x = is_date_data_complete(os.path.join(data_path, '20260412'), 20260412)
    print(x)
