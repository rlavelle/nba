from typing import Tuple

import numpy as np
import pandas as pd

from src.db.db_manager import DBManager
from src.types.game_types import StatType, SeasonType
from src.types.player_types import PlayerType, PlayerFeaturesDF


# todo: clean up block comments to be uniform style
class NBADataLoader:
    def __init__(self):
        self.dbm = DBManager()
        self.data = {}
        self.loaded = False

    def load_data(self,
                  stat_type: StatType = StatType.TOTAL,
                  ssns: Tuple[SeasonType] = (SeasonType.REGULAR,)):
        """
        Load and process NBA game and player data.

        Args:
            stat_type: Type of statistics to load (default: total statistics)
            ssns: Tuple of season types to include (default: regular season only)
        """
        game_stats, game_meta, games = self._load_games(stat_type, ssns)
        self.data['game_stats'] = game_stats
        self.data['game_meta'] = game_meta
        self.data['games'] = games

        player_stats, player_meta, players = self._load_players()
        self.data['player_stats'] = player_stats
        self.data['player_meta'] = player_meta

        # Process and merge the data
        processed_players: PlayerFeaturesDF = self._process_players(players, games)
        self.data['players'] = processed_players

        self.loaded = True

    def _load_games(self,
                    stat_type: StatType,
                    ssns: Tuple[SeasonType]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load game statistics and metadata.
        """
        game_stats = self.dbm.get_game_stats()
        game_meta = self.dbm.get_games()
        season_codes = [s.value for s in ssns]
        game_meta = game_meta[game_meta.season_type_code.isin(season_codes)].copy()
        games = pd.merge(game_meta, game_stats, how='left', on='game_id')
        games = games[games.stat_type == stat_type.value.upper()] #todo: prob should fix this upper nonsense
        return game_stats, game_meta, games

    def _load_players(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load player statistics and metadata.
        """
        player_stats = self.dbm.get_player_stats()
        player_meta = self.dbm.get_players()
        players = pd.merge(player_stats, player_meta, how='left', on='player_id')
        return player_stats, player_meta, players

    def _process_players(self,
                         players: pd.DataFrame,
                         games: pd.DataFrame) -> PlayerFeaturesDF:
        """
        Process player data including calculating minutes, player types, and spreads.
        """
        x = games[['game_id', 'team_id', 'season', 'season_type', 'season_type_code',
                   'dint', 'date', 'is_home']]

        players = pd.merge(x, players, on=['game_id', 'team_id'], how='left')

        tmp = pd.merge(games[games.is_home == 1][['game_id', 'points']],
                       games[games.is_home == 0][['game_id', 'points']],
                       on='game_id', how='left')
        tmp['spread'] = np.abs(tmp.points_x - tmp.points_y)
        players = pd.merge(players, tmp[['game_id', 'spread']], on='game_id', how='left')

        mins = players.groupby(['team_id', 'season', 'player_id']).minutes.agg(
            mu_m='mean', var_m='var').reset_index(drop=False)
        mins = mins.sort_values(by=['team_id', 'season', 'mu_m'])

        mins['rk'] = mins.groupby([mins.team_id, mins.season]).mu_m.rank(
            method='first', ascending=False)

        # Classify player types based on minutes played
        # generally starters play more than 28min (forced top 5 below if not)
        # after that primary bench gets around 20-28, secondary bench 10-20
        # deep bench gets garbage minutes
        mins['player_type'] = np.where(
            mins.mu_m >= 28, PlayerType.STARTER_PLUS.value,
            np.where((mins.mu_m >= 20) & (mins.mu_m < 28), PlayerType.PRIMARY_BENCH.value,
                     np.where((mins.mu_m >= 10) & (mins.mu_m < 20),
                              PlayerType.SECONDARY_BENCH.value,
                              PlayerType.BENCH.value)))

        mins.loc[mins.rk <= 5, 'player_type'] = PlayerType.STARTER.value
        players = pd.merge(players, mins, on=['player_id', 'team_id', 'season'], how='left')
        players['ppm'] = np.where(players.minutes == 0, 0, players.points / players.minutes)

        return players

    def get_player_type(self, ptypes: Tuple[PlayerType]) -> pd.DataFrame:
        """
        Get a specific player type from processed players
        :param ptypes: One of PlayerTypeLiteral
        :return: dataframe subset of player type
        """
        assert self.loaded, "data not loaded"
        players = self.data['players']

        return players[players.player_type.isin(p.value for p in ptypes)].copy()

    def get_data(self, key: str) -> pd.DataFrame:
        """
        Get a specific dataset by key.

        Args:
            key: One of 'game_stats', 'game_meta', 'games', 'player_stats', 'player_meta', 'players'

        Returns:
            Requested DataFrame
        """
        assert self.loaded, "data not loaded"
        return self.data.get(key)
