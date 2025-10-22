import json

from src.scrapers.api import API


class NBAStatsApi(API):
    def __init__(self, logger=None):
        super().__init__(logger)

    def get_boxscore(self, game_id:str, endpoint:str, period:str) -> dict[str]:
        assert endpoint in self.config.options('NBA_STATS_ENDPOINTS'), f'{endpoint} not in valid endpoints'
        assert period in self.config.options('TIME_PERIODS'), f'{period} not in valid time periods'

        url = self.config.get('NBA_STATS_ENDPOINTS', endpoint)
        headers = json.loads(self.config.get('NBA_STATS_HEADERS', 'headers'))
        params = json.loads(self.config.get('TIME_PERIODS', period))
        params['GameID'] = game_id

        response = self._get(url=url, params=params, headers=headers)
        return response

    def get_games(self, date:str) -> dict[str]:
        url = self.config.get('NBA_GAMES_ENDPOINTS', 'nba_games')
        headers = json.loads(self.config.get('NBA_GAMES_HEADERS', 'headers'))
        params = {
            'gamedate': date,
            'platform': 'web'
        }
        response = self._get(url=url, params=params, headers=headers)

        if 'error' in response:
            return response

        if len(response['modules']) == 0:
            return {}

        return response
