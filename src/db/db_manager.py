import numpy as np
import pandas as pd

from src.db.utils import get_engine
from src.logging.logger import Logger
from src.types.game_types import StatType, SeasonType


class DBManager:
    def __init__(self, logger=None):
        self.engine = get_engine()

        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            ret = pd.read_sql_query(query, self.engine)
            if 'date' in ret.columns:
                ret['date'] = pd.to_datetime(ret.date)
            return ret

        except Exception as e:
            self.logger.log(f'[QUERY ERROR]: {e}')
            return pd.DataFrame([])

    def get_games(self) -> pd.DataFrame:
        query = 'SELECT * FROM games;'
        return self.execute_query(query)

    def get_teams(self) -> pd.DataFrame:
        query = 'SELECT * FROM teams;'
        return self.execute_query(query)

    def get_players(self) -> pd.DataFrame:
        query = 'SELECT * FROM players;'
        return self.execute_query(query)

    def get_prop_odds(self) -> pd.DataFrame:
        query = 'SELECT * FROM player_props;'
        return self.execute_query(query)

    def get_spread_odds(self) -> pd.DataFrame:
        query = 'SELECT * FROM game_spreads;'
        return self.execute_query(query)

    def get_money_line_odds(self) -> pd.DataFrame:
        query = 'SELECT * FROM game_ml;'
        return self.execute_query(query)

    def get_prop_results(self) -> pd.DataFrame:
        query = 'SELECT * FROM player_prop_results;'
        return self.execute_query(query)

    def get_money_line_results(self) -> pd.DataFrame:
        query = 'SELECT * FROM game_ml_results;'
        return self.execute_query(query)

    def get_all_player_ids(self) -> np.ndarray:
        query = 'SELECT player_id FROM players;'
        return self.execute_query(query).player_id.values

    def get_all_team_ids(self) -> np.ndarray:
        query = 'SELECT team_id FROM teams;'
        return self.execute_query(query).team_id.values

    def get_player_stats(self) -> pd.DataFrame:
        query = 'SELECT * FROM player_stats;'
        return self.execute_query(query)

    def get_game_stats(self) -> pd.DataFrame:
        query = 'SELECT * FROM game_stats;'
        return self.execute_query(query)

    def get_game_stats_by_type(self, stat_type: StatType) -> pd.DataFrame:
        query = f"SELECT * FROM game_stats WHERE stat_type = '{stat_type.value}';"
        return self.execute_query(query)

    def get_player_id(self, player_slug: str) -> int:
        query = f"SELECT * FROM players WHERE player_slug = '{player_slug}';"
        player = self.execute_query(query)

        if len(player) == 0:
            raise ValueError(f'{player_slug} not found in DB')

        return player['player_id'].values[0]

    def get_team_id(self, team_name: str) -> int:
        query = f"SELECT * FROM teams WHERE team_name = '{team_name}';"
        team = self.execute_query(query)

        if len(team) == 0:
            raise ValueError(f'{team_name} not found in DB')

        return team['team_id'].values[0]

    def get_player_stats_by_slug(self, player_slug: str) -> pd.DataFrame:
        query = f"SELECT * FROM player_stats WHERE player_id = '{self.get_player_id(player_slug)}';"
        return self.execute_query(query)

    def get_player_stats_team(self, player_slug: str, team_name: str) -> pd.DataFrame:
        query = f"""
            SELECT DISTINCT player_stats.*, teams.team_name, players.player_slug
            FROM player_stats
            JOIN teams ON player_stats.team_id = teams.team_id
            JOIN players ON player_stats.player_id = players.player_id
            WHERE players.player_id = '{self.get_player_id(player_slug)}'
            AND teams.team_id = '{self.get_team_id(team_name)}';
        """
        return self.execute_query(query).sort_values(by='date')

    def get_player_stats_season_type(self, player_slug: str, season_type_code: SeasonType,
                                     season: str = None) -> pd.DataFrame:
        query = """
            SELECT DISTINCT s.*, 
                            g.season, g.season_type, g.season_type_code, 
                            g.dint, g.date, g.home_team, g.away_team, 
                            t.team_name, p.player_slug
            FROM player_stats s
            INNER JOIN games g ON s.game_id = g.game_id
            INNER JOIN teams t ON s.team_id = t.team_id
            INNER JOIN players p ON s.player_id = p.player_id
            WHERE s.player_id = '{player_id}'
        """.format(player_id=self.get_player_id(player_slug))

        if season is not None:
            query += f" AND g.season = '{season}'"

        query += f" AND g.season_type_code = '{season_type_code.value}';"

        ret = self.execute_query(query).sort_values(by='date')

        return ret.loc[:, ret.columns[~ret.columns.duplicated()]]

    def get_team_games_season_type(self, team_name: str, season_type_code: SeasonType,
                                   season: str = None) -> pd.DataFrame:
        query = """
            SELECT DISTINCT g.*
            FROM games g
            INNER JOIN teams t ON g.home_team = t.team_id OR g.away_team = t.team_id
            WHERE t.team_id = '{team_id}'
        """.format(team_id=self.get_team_id(team_name))

        if season is not None:
            query += f" AND g.season = '{season}'"

        query += f" AND g.season_type_code = '{season_type_code.value}';"

        res = self.execute_query(query)

        teams = self.get_teams()

        ret = pd.merge(res, teams, left_on='home_team', right_on='team_id', suffixes=('_home', '_away'))
        ret = pd.merge(ret, teams, left_on='away_team', right_on='team_id', suffixes=('_home', '_away'))

        return ret.drop(['team_id_home', 'team_id_away'], axis=1).sort_values(by='date')

    def get_stats_season_type(self, season_type_code: SeasonType, season: str = None) -> pd.DataFrame:
        query = f"""
            SELECT s.*, g.season, g.season_type, g.season_type_code, g.dint, g.date
            FROM player_stats s
            INNER JOIN games g ON s.game_id = g.game_id
            WHERE g.season_type_code = '{season_type_code.value}'
        """
        if season is not None:
            query += f" AND g.season = '{season}'"

        return self.execute_query(query)

    def get_games_with_teams(self, dint: int) -> pd.DataFrame:
        query = f"""
            SELECT
                g.game_id,
                g.season,
                g.season_type,
                g.season_type_code,
                g.dint,
                g.date,
                gs_home.team_id AS home_team_id,
                gs_away.team_id AS away_team_id
            FROM games g
            LEFT JOIN game_stats gs_home
                ON g.game_id = gs_home.game_id AND gs_home.is_home = 1
            LEFT JOIN game_stats gs_away
                ON g.game_id = gs_away.game_id AND gs_away.is_home = 0
            WHERE g.dint = {dint};
        """
        return self.execute_query(query)
