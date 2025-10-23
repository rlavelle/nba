from enum import Enum
from typing import Literal

GAME_FEATURES = [
    'minutes', 'pointsOffTurnovers',
    'pointsSecondChance', 'pointsFastBreak', 'pointsPaint',
    'oppPointsOffTurnovers', 'oppPointsSecondChance', 'oppPointsFastBreak',
    'oppPointsPaint', 'fieldGoalsMade', 'fieldGoalsAttempted',
    'fieldGoalsPercentage', 'threePointersMade', 'threePointersAttempted',
    'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted',
    'freeThrowsPercentage', 'reboundsOffensive', 'reboundsDefensive',
    'reboundsTotal', 'assists', 'steals', 'blocks', 'blocksAgainst',
    'turnovers', 'foulsPersonal', 'foulsDrawn', 'points', 'plusMinusPoints',
    'effectiveFieldGoalPercentage', 'trueShootingPercentage',
    'usagePercentage', 'estimatedUsagePercentage', 'offensiveRating',
    'estimatedOffensiveRating', 'defensiveRating',
    'estimatedDefensiveRating', 'netRating', 'estimatedNetRating',
    'assistPercentage', 'assistToTurnover', 'assistRatio', 'turnoverRatio',
    'teamTurnoverPercentage', 'offensiveReboundPercentage',
    'defensiveReboundPercentage', 'reboundPercentage',
    'freeThrowAttemptRate', 'pace', 'estimatedPace', 'pacePer40',
    'possessions', 'PIE', 'oppEffectiveFieldGoalPercentage',
    'oppFreeThrowAttemptRate', 'oppTeamTurnoverPercentage',
    'oppOffensiveReboundPercentage', 'percentageFieldGoalsAttempted2pt',
    'percentageFieldGoalsAttempted3pt', 'percentagePoints2pt',
    'percentagePointsMidrange2pt', 'percentagePoints3pt',
    'percentagePointsFastBreak', 'percentagePointsFreeThrow',
    'percentagePointsOffTurnovers', 'percentagePointsPaint',
    'percentageAssisted2pt', 'percentageUnassisted2pt',
    'percentageAssisted3pt', 'percentageUnassisted3pt',
    'percentageAssistedFGM', 'percentageUnassistedFGM'
]

CURRENT_SEASON = '2025-26'

SeasonTypeLiteral = Literal[
    "Regular Season", "Playoffs", "All-Star", "Preseason", "Summer League", "PlayIn", "IST Championship"]
StatTypeLiteral = Literal["starters", "bench", "statistics"]


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
