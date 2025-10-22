import datetime
from typing import List

from src.scrapers.api import API
from src.types.odds_types import UpcomingGameResponse, EventOdds


class OddsApi(API):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.key = self.config.get('ODDS_API', 'key')
        self.url = self.config.get('ODDS_API', 'base_url')
        self.sport = 'basketball_nba'
        self.regions = 'us'
        self.oddsFormat = 'decimal'

        # TODO: this is temporary, to save credits and to get better true
        #       opening lines we should get games as far ahead as possible
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        self.commenceTimeTo = tomorrow

    def get_spread_ml(self) -> list[EventOdds]:
        markets = 'h2h,spreads'
        params = dict(
            markets=markets,
            regions=self.regions,
            oddsFormat=self.oddsFormat,
            commenceTimeTo=self.commenceTimeTo,
            apiKey=self.key
        )

        url = f'{self.url}/{self.sport}/odds/?'

        return self._get(url=url, params=params, headers=None)

    def get_props(self, id:str) -> EventOdds:
        markets = 'player_points'
        params = dict(
            markets=markets,
            regions=self.regions,
            oddsFormat=self.oddsFormat,
            commenceTimeTo=self.commenceTimeTo,
            apiKey=self.key
        )

        url = f'{self.url}/{self.sport}/events/{id}/odds/?'

        return self._get(url=url, params=params, headers=None)

    def get_upcoming_games(self) -> List[UpcomingGameResponse]:
        params = dict(
            apiKey=self.key,
            commenceTimeTo=self.commenceTimeTo
        )

        url = f'{self.url}/{self.sport}/events/?'

        return self._get(url=url, params=params)