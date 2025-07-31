from datetime import datetime
from enum import Enum
from typing import Literal, TypedDict, NewType

import pandas as pd

from src.types.game_types import SeasonType

PLAYER_FEATURES = [
    'minutes', 'fieldGoalsMade', 'ppm',
    'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade',
    'threePointersAttempted', 'threePointersPercentage', 'freeThrowsMade',
    'freeThrowsAttempted', 'freeThrowsPercentage', 'reboundsOffensive',
    'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks',
    'blocksAgainst', 'turnovers', 'foulsPersonal', 'foulsDrawn', 'points',
    'plusMinusPoints', 'offensiveRating', 'estimatedOffensiveRating',
    'defensiveRating', 'estimatedDefensiveRating', 'netRating',
    'estimatedNetRating', 'assistPercentage', 'assistToTurnover',
    'assistRatio', 'turnoverRatio', 'offensiveReboundPercentage',
    'defensiveReboundPercentage', 'reboundPercentage',
    'effectiveFieldGoalPercentage', 'trueShootingPercentage',
    'usagePercentage', 'estimatedUsagePercentage', 'pace', 'estimatedPace',
    'pacePer40', 'possessions', 'PIE', 'percentageFieldGoalsAttempted2pt',
    'percentageFieldGoalsAttempted3pt', 'percentagePoints2pt',
    'percentagePointsMidrange2pt', 'percentagePoints3pt',
    'percentagePointsFastBreak', 'percentagePointsFreeThrow',
    'percentagePointsOffTurnovers', 'percentagePointsPaint',
    'freeThrowAttemptRate', 'percentageAssisted2pt',
    'percentageUnassisted2pt', 'percentageAssisted3pt',
    'percentageUnassisted3pt', 'percentageAssistedFGM',
    'percentageUnassistedFGM', 'pointsOffTurnovers', 'pointsSecondChance',
    'pointsFastBreak', 'pointsPaint', 'oppPointsOffTurnovers',
    'oppPointsSecondChance', 'oppPointsFastBreak', 'oppPointsPaint',
    'oppEffectiveFieldGoalPercentage', 'oppFreeThrowAttemptRate',
    'oppTeamTurnoverPercentage', 'oppOffensiveReboundPercentage',
    'teamTurnoverPercentage', 'percentagePersonalFouls',
    'percentagePersonalFoulsDrawn'
]

PlayerTypeLiteral = Literal["S", "S1", "PB", "SB", "B"]


class PlayerType(str, Enum):
    STARTER = 'S'
    STARTER_PLUS = 'S1'
    PRIMARY_BENCH = 'PB'
    SECONDARY_BENCH = 'SB'
    BENCH = 'B'


class PlayerFeatures(TypedDict):
    player_id: int
    player_slug: str
    team_id: int
    game_id: str

    season: str
    season_type_code: SeasonType
    date: datetime

    is_home: bool

    minutes: float
    fieldGoalsMade: float
    ppm: float
    fieldGoalsAttempted: float
    fieldGoalsPercentage: float
    threePointersMade: float
    threePointersAttempted: float
    threePointersPercentage: float
    freeThrowsMade: float
    freeThrowsAttempted: float
    freeThrowsPercentage: float
    reboundsOffensive: float
    reboundsDefensive: float
    reboundsTotal: float
    assists: float
    steals: float
    blocks: float
    blocksAgainst: float
    turnovers: float
    foulsPersonal: float
    foulsDrawn: float
    points: float
    plusMinusPoints: float
    offensiveRating: float
    estimatedOffensiveRating: float
    defensiveRating: float
    estimatedDefensiveRating: float
    netRating: float
    estimatedNetRating: float
    assistPercentage: float
    assistToTurnover: float
    assistRatio: float
    turnoverRatio: float
    offensiveReboundPercentage: float
    defensiveReboundPercentage: float
    reboundPercentage: float
    effectiveFieldGoalPercentage: float
    trueShootingPercentage: float
    usagePercentage: float
    estimatedUsagePercentage: float
    pace: float
    estimatedPace: float
    pacePer40: float
    possessions: float
    PIE: float
    percentageFieldGoalsAttempted2pt: float
    percentageFieldGoalsAttempted3pt: float
    percentagePoints2pt: float
    percentagePointsMidrange2pt: float
    percentagePoints3pt: float
    percentagePointsFastBreak: float
    percentagePointsFreeThrow: float
    percentagePointsOffTurnovers: float
    percentagePointsPaint: float
    freeThrowAttemptRate: float
    percentageAssisted2pt: float
    percentageUnassisted2pt: float
    percentageAssisted3pt: float
    percentageUnassisted3pt: float
    percentageAssistedFGM: float
    percentageUnassistedFGM: float
    pointsOffTurnovers: float
    pointsSecondChance: float
    pointsFastBreak: float
    pointsPaint: float
    oppPointsOffTurnovers: float
    oppPointsSecondChance: float
    oppPointsFastBreak: float
    oppPointsPaint: float
    oppEffectiveFieldGoalPercentage: float
    oppFreeThrowAttemptRate: float
    oppTeamTurnoverPercentage: float
    oppOffensiveReboundPercentage: float
    teamTurnoverPercentage: float
    percentagePersonalFouls: float
    percentagePersonalFoulsDrawn: float


PlayerFeaturesDF = NewType('PlayerFeaturesDF', pd.DataFrame)
