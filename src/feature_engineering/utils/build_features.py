import pickle
import time
from typing import Tuple

import pandas as pd

from src.feature_engineering.base import FeaturePipeline, BaseFeature
from src.feature_engineering.bayes_posterior import BayesPosteriorFeature
from src.feature_engineering.last_game_value import LastGameValueFeature
from src.feature_engineering.moving_avg import ExponentialMovingAvgFeature, CumSeasonAvgFeature, CumSeasonEMAFeature, \
    SimpleMovingAvgFeature
from src.feature_engineering.player_streak import PlayerHotStreakFeature
from src.feature_engineering.utils.cache import gen_cache_file, check_cache
from src.logging.logger import Logger
from src.modeling_framework.framework.constants import FS_MINUTES, FS_POINTS
from src.modeling_framework.framework.dataloader import NBADataLoader
from src.types.game_types import GAME_FEATURES, CURRENT_SEASON
from src.types.player_types import PlayerType, PLAYER_FEATURES
from src.utils.date import dint_to_date


def get_ft_cols(data: list[BaseFeature] | pd.DataFrame):
    """
    Works with either a pandas DataFrame or a list of Feature objects
    """
    wanted_subs = ["_bayes_post", "_1g", "_sma_", "_ema_", "_cum_ssn_", "_hot_streak"]

    # Case 1: It's a pandas DataFrame
    if hasattr(data, 'columns'):  # Duck typing for DataFrame
        cols = [
            col for col in data.columns
            if any(sub in col for sub in wanted_subs)
        ]
        return cols

    # Case 2: It's a list of Feature objects
    elif isinstance(data, list):
        cols = [
            feat.feature_name for feat in data
            if any(sub in feat.feature_name for sub in wanted_subs)
        ]
        return cols

    # Case 3: Invalid type
    else:
        raise TypeError(f"Expected DataFrame or list of Feature objects, got {type(data)}")


def is_child_feature(f: BaseFeature, dependents: list[BaseFeature]):
    return any(f.feature_name in d.feature_name for d in dependents)


def get_ft_sets(fts, id) -> Tuple[list[BaseFeature], list[BaseFeature]]:
    full_ft_set = []
    full_dependents_set = []

    for f in fts:
        features = [
            ExponentialMovingAvgFeature(span=7, source_col=f, group_col=(id,)),
            ExponentialMovingAvgFeature(span=5, source_col=f, group_col=(id,)),
            ExponentialMovingAvgFeature(span=3, source_col=f, group_col=(id,)),
            CumSeasonAvgFeature(source_col=f, group_col=(id, 'season')),
            CumSeasonEMAFeature(source_col=f, group_col=(id, 'season')),
            SimpleMovingAvgFeature(window=3, source_col=f, group_col=(id,)),
            SimpleMovingAvgFeature(window=5, source_col=f, group_col=(id,)),
            LastGameValueFeature(source_col=f, group_col=(id,))
        ]

        dependents = []
        for feature in features:
            if '_cum_ssn' in feature.feature_name:
                dependents.append(
                    BayesPosteriorFeature(ybar_col=feature.feature_name,
                                          source_col=f,
                                          id_col=id,
                                          group_col=(id, 'season'))
                )

            if '3g' not in feature.feature_name and '1g' not in feature.feature_name:
                dependents.append(
                    PlayerHotStreakFeature(window=1,
                                           comp_col=feature.feature_name,
                                           source_col=f,
                                           group_col=(id, 'season'))
                )

                dependents.append(
                    PlayerHotStreakFeature(window=3,
                                           comp_col=feature.feature_name,
                                           source_col=f,
                                           group_col=(id, 'season'))
                )

            if '3g' in feature.feature_name:
                dependents.append(
                    PlayerHotStreakFeature(window=1,
                                           comp_col=feature.feature_name,
                                           source_col=f,
                                           group_col=(id, 'season'))
                )

        full_dependents_set.extend(dependents)
        full_ft_set.extend(features)

    return full_ft_set, full_dependents_set


def build_game_lvl_fts(logger: Logger = None, cache: bool = False, date: int = None, recent: bool = False):
    start = time.time()

    if logger:
        logger.log(f'[STARTING GAME FEATURE BUILDING]')

    if cache:
        if logger:
            ret = check_cache(f='game_fts', logger=logger, date=date, recent=recent)
        else:
            ret = check_cache(f='game_fts', date=date, recent=recent)

        if ret is not None:
            # when pulling from recent cache need to remove future data,
            # but keep nulled future rows
            # TODO: somehow there is model bleed, predictions are not the same
            #       as when they are run without --recent
            if date and recent:
                ret = ret[ret.date <= dint_to_date(date)].copy()
                ret.loc[ret.date == dint_to_date(date), 'date'] = pd.Timestamp('2030-01-01')

            return ret

    data_loader = NBADataLoader()
    data_loader.load_data()

    if logger:
        t = time.time()
        logger.log(f'[DATA LOADED]: {round((t - start), 2)}s')

    games = data_loader.get_data('games')
    games = games.dropna()
    games = games.sort_values(by=['team_id', 'season', 'date'])

    # for predictions, for each team we add a null row
    #   - row contains the team id, curr season, and a date of 2030-01-01
    teams = games[['team_id']].drop_duplicates()
    future_rows = pd.DataFrame(columns=games.columns)

    future_rows['team_id'] = teams['team_id']
    future_rows['season'] = CURRENT_SEASON
    future_rows['date'] = pd.Timestamp('2030-01-01')

    games = pd.concat([games, future_rows], ignore_index=True)

    features, dependents = get_ft_sets(GAME_FEATURES, 'team_id')
    fts_cols = get_ft_cols(features)
    dps_cols = get_ft_cols(dependents)

    features = [f for f in features if f.feature_name in fts_cols]
    dependents = [d for d in dependents if d.feature_name in dps_cols]

    if logger:
        t1 = time.time()
        logger.log(f'[FEATURE SETS CREATED]: {round((t1 - t), 2)}s')

    # need to build the raw features first as the dependents are based on these
    pipeline = FeaturePipeline(features, logger=logger)
    games = pipeline.transform(games, sort_order=('game_id', 'season', 'date'))

    pipeline = FeaturePipeline(dependents, logger=logger)
    games = pipeline.transform(games, sort_order=('game_id', 'season', 'date'))

    games = games.dropna(subset=fts_cols + dps_cols)
    games = games[games['season'] > games['season'].min()].copy()

    if logger:
        end = time.time()
        logger.log(f'[FEATURE BUILDING COMPLETE]: {round((end - t1), 2)}s')

    try:
        with open(gen_cache_file('game_fts', date), "wb") as f:
            pickle.dump(games, f)
    except Exception as e:
        if logger:
            logger.log(f'[ERROR ON SAVING CACHE]: {e}')

    return games


def build_player_lvl_fts(logger: Logger = None, cache: bool = False, date: int = None, recent: bool = False):
    start = time.time()

    if logger:
        logger.log(f'[STARTING PLAYER FEATURE BUILDING]')

    if cache:
        if logger:
            ret = check_cache(f='player_fts', logger=logger, date=date, recent=recent)
        else:
            ret = check_cache(f='player_fts', date=date, recent=recent)

        if ret is not None:
            # when pulling from recent cache need to remove future data,
            # but keep nulled future rows
            if date and recent:
                ret = ret[ret.date <= dint_to_date(date)].copy()
                ret.loc[ret.date == dint_to_date(date), 'date'] = pd.Timestamp('2030-01-01')

            return ret

    data_loader = NBADataLoader()
    data_loader.load_data()

    if logger:
        t = time.time()
        logger.log(f'[DATA LOADED]: {round((t - start), 2)}s')

    ptypes = (PlayerType.STARTER, PlayerType.STARTER_PLUS, PlayerType.PRIMARY_BENCH)
    player_data = data_loader.get_player_type(ptypes=ptypes)
    player_data = player_data[~player_data.spread.isna()].copy()
    player_data = player_data.drop(columns=['position'])
    player_data = player_data.dropna()
    player_data = player_data.sort_values(by=['player_id', 'season', 'date'])

    # for predictions, for each player we add a null row
    #   - row contains the player id, curr season, and a date of 2030-01-01
    players = player_data[['player_id']].drop_duplicates()
    future_rows = pd.DataFrame(columns=player_data.columns)

    future_rows['player_id'] = players['player_id']
    future_rows['season'] = CURRENT_SEASON
    future_rows['date'] = pd.Timestamp('2030-01-01')

    player_data = pd.concat([player_data, future_rows], ignore_index=True)

    features, dependents = get_ft_sets(PLAYER_FEATURES, 'player_id')

    # only need the features that actually build the minutes + ppm model
    # but, we need to retain the features that appear in the dependents
    dependents = [d for d in dependents if d.feature_name in list(set(FS_MINUTES + FS_POINTS))]
    features = [f for f in features if is_child_feature(f, dependents) or f.feature_name in list(set(FS_MINUTES + FS_POINTS))]

    if logger:
        t1 = time.time()
        logger.log(f'[FEATURE SETS CREATED]: {round((t1 - t), 2)}s')

    # build out features then dependents
    pipeline = FeaturePipeline(features, logger=logger)
    player_data = pipeline.transform(player_data, sort_order=('player_id', 'season', 'date'))

    pipeline = FeaturePipeline(dependents, logger=logger)
    player_data = pipeline.transform(player_data, sort_order=('player_id', 'season', 'date'))

    player_data = player_data.dropna(subset=get_ft_cols(features + dependents))
    player_data = player_data[player_data['season'] > player_data['season'].min()].copy()

    if logger:
        end = time.time()
        logger.log(f'[FEATURE BUILDING COMPLETE]: {round((end - t1), 2)}s')

    try:
        with open(gen_cache_file('player_fts', date), "wb") as f:
            pickle.dump(player_data, f)
    except Exception as e:
        if logger:
            logger.log(f'[ERROR ON SAVING CACHE]: {e}')

    return player_data
