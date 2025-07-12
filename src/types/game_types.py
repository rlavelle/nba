from enum import Enum
from typing import Literal

SeasonTypeLiteral = Literal["Regular Season", "Playoffs", "All-Star", "Preseason", "Summer League", "PlayIn", "IST Championship"]


class StrEnum(str, Enum):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

class StatType(StrEnum):
    STARTERS = "starters"
    BENCH = "bench"
    TOTAL = "statistics"

class SeasonType(StrEnum):
    REGULAR = '00'
    PLAYOFFS = '01'
    ALL_STAR = '02'
    PRESEASON = '03'
    SUMMER_LEAGUE = '04'
    PLAY_IN = '05'
    IST_CHAMPIONSHIP = '06'