from enum import Enum
from typing import Literal


SeasonTypeLiteral = Literal["Regular Season", "Playoffs", "All-Star", "Preseason", "Summer League", "PlayIn", "IST Championship"]


class StatType(str, Enum):
    STARTERS = "starters"
    BENCH = "bench"
    TOTAL = "statistics"


class SeasonType(str, Enum):
    REGULAR = '00'
    PLAYOFFS = '01'
    ALL_STAR = '02'
    PRESEASON = '03'
    SUMMER_LEAGUE = '04'
    PLAY_IN = '05'
    IST_CHAMPIONSHIP = '06'