import configparser
import json
import src.api.api_utils as util
from src.config import CONFIG_PATH

class NBAStatsApi:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)

    def get_boxscore(self, game_id:str, endpoint:str, period:str) -> dict[str]:
        assert endpoint in self.config.options('NBA_STATS_ENDPOINTS'), f'{endpoint} not in valid endpoints'
        assert period in self.config.options('TIME_PERIODS'), f'{period} not in valid time periods'

        url = self.config.get('NBA_STATS_ENDPOINTS', endpoint)
        headers = json.loads(self.config.get('NBA_STATS_HEADERS', 'headers'))
        params = json.loads(self.config.get('TIME_PERIODS', period))
        params['GameID'] = game_id

        response = util.post(url=url, params=params, headers=headers)
        return response

    def get_games(self, date:str) -> dict[str]:
        url = self.config.get('NBA_GAMES_ENDPOINTS', 'nba_games')
        headers = json.loads(self.config.get('NBA_GAMES_HEADERS', 'headers'))
        params = {
            'gamedate': date,
            'platform': 'web'
        }
        response = util.post(url=url, params=params, headers=headers)

        if 'error' in response:
            return response

        if len(response['modules']) == 0:
            return {}

        return response


