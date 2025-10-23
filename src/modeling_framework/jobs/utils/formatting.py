import pandas as pd

from src.feature_engineering.utils.build_features import build_game_lvl_fts, build_player_lvl_fts
from src.logging.logger import Logger


def fmt_diff_data(logger: Logger = None, cache: bool = False):
    data = build_game_lvl_fts(logger=logger, cache=cache)

    meta_cols = [
        "season", "season_type", "season_type_code", "dint",
        "date", "team_id", "is_home", "stat_type"
    ]

    home = data[data["is_home"] == 1].set_index("game_id")
    away = data[data["is_home"] == 0].set_index("game_id")
    meta = home[meta_cols].reset_index()

    home_stats = home.drop(columns=meta_cols)
    away_stats = away.drop(columns=meta_cols)

    diff_stats = home_stats - away_stats
    diff_stats = diff_stats.reset_index()

    df_diff = pd.concat([meta, diff_stats.drop(columns=["game_id"])], axis=1)
    return df_diff


def fmt_player_data(logger: Logger = None, cache: bool = False):
    data = build_player_lvl_fts(logger=logger, cache=cache)
    data = data[(data.points > 0) | (data.points.isna())].copy()

    data = data.sort_values(by=['player_id', 'season', 'date'])

    data['ppm_s1'] = (
        data.groupby(['player_id', 'season'])['ppm']
        .expanding()
        .mean()
        .shift(1)
        .fillna(0)
        .reset_index(drop=True)
        .values
    )
    data['ppm_diff'] = data.ppm - data.ppm_s1
    return data
