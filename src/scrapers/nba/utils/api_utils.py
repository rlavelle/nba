import json
import os

from src.logging.logger import Logger
from src.scrapers.nba.nba_stats_api import NBAStatsApi
from src.scrapers.nba.utils.parsing import parse_boxscore
from src.scrapers.nba.utils.validation import stat_type_exists, is_bad_stat


def fetch_and_save_boxscore(game_id: str,
                            boxscore: str,
                            api: NBAStatsApi,
                            data_path: str,
                            logger: Logger):
    stat_type_fpath = os.path.join(data_path, f'{game_id}_{boxscore}_stats.json')
    if stat_type_exists(stat_type_fpath):
        return

    score = api.get_boxscore(game_id=game_id, endpoint=boxscore, period='full_game')

    if 'error' in score:
        print(f'bad api hit on {game_id}')
        logger.log(f'bad api hit on {game_id}')
        logger.log(score['error'])
        raise Exception(f'bad api hit on {game_id}: {score["error"]}')

    validate_game = score['meta']['request'].split('/')[4]
    assert game_id == validate_game, f'{game_id} <> {validate_game}'

    fmt_game = parse_boxscore(score=score)
    if is_bad_stat(fmt_game):
        logger.log(f'[EMPTY GAME DATA] {game_id} {stat_type_fpath}')
        return

    json.dump(fmt_game, open(stat_type_fpath, 'w'))

    msg = f"[SUCCESS] {game_id}/{boxscore}"
    logger.log(msg)
