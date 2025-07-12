from datetime import datetime
from typing import TypedDict, Any

from src.types.game_types import SeasonTypeLiteral, SeasonType, StatType


class RawPlayerData(TypedDict):
    personId: int
    firstName: str
    familyName: str
    playerSlug: str
    statistics: dict[str, Any]
    position: str


class RawGameMeta(TypedDict):
    meta: Any
    home: dict[str, Any]
    away: dict[str, Any]


class RawTeamData(TypedDict):
    teamId: int
    teamName: str
    teamTricode: str
    players: list[RawPlayerData]
    statistics: dict[str, Any]
    bench: dict[str, Any]
    starters: dict[str, Any]


class RawGameStats(TypedDict):
    gameId: str
    awayTeamId: int
    homeTeamId: int
    homeTeam: RawTeamData
    awayTeam: RawTeamData


class PlayerMeta(TypedDict):
    player_id: int
    player_name: str
    player_slug: str


class PlayerStats(TypedDict, total=False):
    player_id: int
    team_id: int
    game_id: str
    position: str
    minutes: float
    # all auxiliary stats,,,


class TeamMeta(TypedDict):
    team_id: int
    team_name: str
    team_slug: str


class GameMeta(TypedDict):
    game_id: str
    season: str
    season_type: SeasonTypeLiteral
    season_type_code: SeasonType
    dint: int
    date: datetime.date


class GameStats(TypedDict, total=False):
    game_id: str
    team_id: int
    is_home: bool
    stat_type: StatType
    # all auxiliary stats...


