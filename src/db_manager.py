import pandas as pd


class DBManager:
    def __init__(self, engine):
        self.engine = engine

    def execute_query(self, query):
        return pd.read_sql_query(query, self.engine)

    def get_games(self):
        query = 'SELECT * FROM games;'
        return self.execute_query(query)

    def get_teams(self):
        query = 'SELECT * FROM teams;'
        return self.execute_query(query)

    def get_players(self):
        query = 'SELECT * FROM players;'
        return self.execute_query(query)

    def get_stats(self):
        query = 'SELECT * FROM stats;'
        return self.execute_query(query)

    def get_player_id(self, player_slug):
        query = f"SELECT * FROM players WHERE player_slug = '{player_slug}';"
        player = self.execute_query(query)

        if len(player) == 0:
            raise ValueError(f'{player_slug} not found in DB')

        return player['player_id'].values[0]

    def get_team_id(self, team_name):
        query = f"SELECT * FROM teams WHERE team_name = '{team_name}';"
        team = self.execute_query(query)

        if len(team) == 0:
            raise ValueError(f'{team_name} not found in DB')

        return team['team_id'].values[0]

    def get_player_stats(self, player_slug):
        query = f"SELECT * FROM stats WHERE player_id = '{self.get_player_id(player_slug)}';"
        return self.execute_query(query)

    def get_player_stats_team(self, player_slug, team_name):
        query = f"""
            SELECT *
            FROM stats
            WHERE player_id = '{self.get_player_id(player_slug)}'
            AND team_id = '{self.get_team_id(team_name)}';
        """
        return self.execute_query(query)

    def get_player_stats_season(self, player_slug, season_type_code, season=None):
        query = """
            SELECT *
            FROM stats s
            INNER JOIN games g ON s.game_id = g.game_id
            WHERE s.player_id = '{player_id}'
            """.format(player_id=self.get_player_id(player_slug))

        if season is not None:
            query += f" AND g.season = '{season}'"

        query += f" AND g.season_type_code = '{season_type_code}';"

        return self.execute_query(query)

    def get_team_games_season(self, team_name, season_type_code, season=None):
        query = """
            SELECT *
            FROM games
            WHERE (home_team = '{team_id}'
            OR away_team = '{team_id}')
        """.format(team_id=self.get_team_id(team_name))

        if season is not None:
            query += f" AND season = '{season}'"

        query += f" AND season_type_code = '{season_type_code}';"

        return self.execute_query(query)

    def get_teams_season(self, season, season_type_code):
        query = f"""
            SELECT DISTINCT t.*
            FROM teams t
            JOIN (
                SELECT home_team AS team_id
                FROM games
                WHERE season = '{season}'
                AND season_type_code = '{season_type_code}'
                UNION
                SELECT away_team AS team_id
                FROM games
                WHERE season = '{season}'
                AND season_type_code = '{season_type_code}'
            ) AS teams_in_season
            ON t.team_id = teams_in_season.team_id;
        """
        return self.execute_query(query)
