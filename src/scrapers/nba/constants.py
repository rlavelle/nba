from src.types.game_types import SeasonType

N_STAT_TYPES = 7 # number of types of stats pulled per game
SEASON_TYPE_MAP = {
    'Regular Season': SeasonType.REGULAR,
    'Playoffs': SeasonType.PLAYOFFS,
    'All-Star': SeasonType.ALL_STAR,
    'Preseason': SeasonType.PRESEASON,
    'Summer League': SeasonType.SUMMER_LEAGUE,
    'PlayIn': SeasonType.PLAY_IN,
    'IST Championship': SeasonType.IST_CHAMPIONSHIP
}
PLAYER_DUPE_COLS = [
    "percentagePoints",
    "percentageFieldGoalsMade",
    "percentageFieldGoalsAttempted",
    "percentageThreePointersMade",
    "percentageThreePointersAttempted",
    "percentageFreeThrowsMade",
    "percentageFreeThrowsAttempted",
    "percentageReboundsOffensive",
    "percentageReboundsDefensive",
    "percentageReboundsTotal",
    "percentageAssists",
    "percentageTurnovers",
    "percentageSteals",
    "percentageBlocks",
    "percentageBlocksAllowed"
]
GAME_DUPE_COLS = [
    "estimatedTeamTurnoverPercentage"  # Same as teamTurnoverPercentage
]
