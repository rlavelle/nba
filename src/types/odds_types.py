from datetime import datetime
from typing import TypedDict

class UpcomingGameResponse(TypedDict):
    id: str
    sport_key: str
    sport_title: str
    commence_time: datetime
    home_team: str
    away_team: str

class MoneyLineTeamOdds(TypedDict):
    name: str
    price: float

class SpreadTeamOdds(TypedDict):
    name: str
    price: float
    point: float

class PropOdds(TypedDict):
    name: str
    description: str
    price: float
    point: float

class MarketOdds(TypedDict):
    key: str  # h2h, spread, etc.
    outcomes: list[MoneyLineTeamOdds | SpreadTeamOdds]

class BookmakerOdds(TypedDict):
    key: str
    title: str
    last_update: datetime
    markets: list[MarketOdds]

class MarketPropOdds(TypedDict):
    key: str
    last_update: datetime
    outcomes: list[PropOdds]

class BookmakerPropOdds(TypedDict):
    key: str
    title: str
    markets: list[MarketPropOdds]

class EventOdds(TypedDict):
    id: str
    sports_key: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: list[BookmakerOdds|BookmakerPropOdds]