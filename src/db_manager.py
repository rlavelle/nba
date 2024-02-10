import pandas as pd


class DBManager:
    def __init__(self, engine):
        self.engine = engine

    def execute_query(self, query):
        ret = pd.read_sql_query(query, self.engine)
        if 'date' in ret.columns:
            ret['date'] = pd.to_datetime(ret.date)

        return ret

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
            SELECT DISTINCT stats.*, teams.team_name, players.player_slug
            FROM stats
            JOIN teams ON stats.team_id = teams.team_id
            JOIN players ON stats.player_id = players.player_id
            WHERE players.player_id = '{self.get_player_id(player_slug)}'
            AND teams.team_id = '{self.get_team_id(team_name)}';
        """
        return self.execute_query(query).sort_values(by='date')

    def get_player_stats_season(self, player_slug, season_type_code, season=None):
        query = """
            SELECT DISTINCT s.*, 
                            g.season, g.season_type, g.season_type_code, 
                            g.dint, g.date, g.home_team, g.away_team, 
                            g.home_score, g.away_score,
                            t.team_name, p.player_slug
            FROM stats s
            INNER JOIN games g ON s.game_id = g.game_id
            INNER JOIN teams t ON s.team_id = t.team_id
            INNER JOIN players p ON s.player_id = p.player_id
            WHERE s.player_id = '{player_id}'
        """.format(player_id=self.get_player_id(player_slug))

        if season is not None:
            query += f" AND g.season = '{season}'"

        query += f" AND g.season_type_code = '{season_type_code}';"

        ret = self.execute_query(query).sort_values(by='date')

        return ret.loc[:, ret.columns[~ret.columns.duplicated()]]

    def get_team_games_season(self, team_name, season_type_code, season=None):
        query = """
            SELECT DISTINCT g.*
            FROM games g
            INNER JOIN teams t ON g.home_team = t.team_id OR g.away_team = t.team_id
            WHERE t.team_id = '{team_id}'
        """.format(team_id=self.get_team_id(team_name))

        if season is not None:
            query += f" AND g.season = '{season}'"

        query += f" AND g.season_type_code = '{season_type_code}';"

        res = self.execute_query(query)

        teams = self.get_teams()

        ret = pd.merge(res, teams, left_on='home_team', right_on='team_id', suffixes=('_home', '_away'))
        ret = pd.merge(ret, teams, left_on='away_team', right_on='team_id', suffixes=('_home', '_away'))

        return ret.drop(['team_id_home', 'team_id_away'], axis=1).sort_values(by='date')

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
