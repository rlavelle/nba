ERROR_LOG_SCHEMA = """
DROP TABLE IF EXISTS errors;
CREATE TABLE IF NOT EXISTS errors (
    task_id INTEGER NOT NULL,
    game_id TEXT,
    team_id INTEGER,
    player_id INTEGER,
    time TIMESTAMP,
    msg TEXT,
    PRIMARY KEY (task_id)
);
"""

TEAMS_SCHEMA = """
DROP TABLE IF EXISTS teams;
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER NOT NULL,
    team_name TEXT,
    team_slug TEXT,
    PRIMARY KEY (team_id)
);"""

PLAYERS_SCHEMA = """
DROP TABLE IF EXISTS players;
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER NOT NULL,
    player_name TEXT,
    player_slug TEXT,
    PRIMARY KEY (player_id)
);"""

GAMES_META_SCHEMA = """
DROP TABLE IF EXISTS games;
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT NOT NULL,
    season TEXT,
    season_type TEXT,
    season_type_code TEXT,
    dint INTEGER,
    date TIMESTAMP,
    PRIMARY KEY (game_id)
);"""

PLAYER_PROP_ODDS_SCHEMA = """
DROP TABLE IF EXISTS "player_props";
CREATE TABLE IF NOT EXISTS "player_props" (
    "player_id" INTEGER NOT NULL,
    "dint" INTEGER NOT NULL,
    "bookmaker" TEXT,
    "last_update" TIMESTAMP,
    "odd_type" TEXT,
    "description" TEXT,
    "price" FLOAT,
    "point" FLOAT,
    
    PRIMARY KEY ("player_id", "dint", "bookmaker", "last_update", "odd_type", "description", "point")
);
"""

PLAYER_STATS_SCHEMA = """
DROP TABLE IF EXISTS "player_stats";
CREATE TABLE IF NOT EXISTS "player_stats" (
    "player_id" INTEGER NOT NULL,
    "team_id" INTEGER NOT NULL,
    "game_id" TEXT NOT NULL,
    "position" TEXT,

    "minutes" FLOAT,

    "fieldGoalsMade" INTEGER,
    "fieldGoalsAttempted" INTEGER,
    "fieldGoalsPercentage" FLOAT,
    "threePointersMade" INTEGER,
    "threePointersAttempted" INTEGER,
    "threePointersPercentage" FLOAT,
    "freeThrowsMade" INTEGER,
    "freeThrowsAttempted" INTEGER,
    "freeThrowsPercentage" FLOAT,

    "reboundsOffensive" INTEGER,
    "reboundsDefensive" INTEGER,
    "reboundsTotal" INTEGER,

    "assists" INTEGER,
    "steals" INTEGER,
    "blocks" INTEGER,
    "blocksAgainst" INTEGER,
    "turnovers" INTEGER,
    "foulsPersonal" INTEGER,
    "foulsDrawn" INTEGER,

    "points" INTEGER,
    "plusMinusPoints" FLOAT,

    "offensiveRating" FLOAT,
    "estimatedOffensiveRating" FLOAT,
    "defensiveRating" FLOAT,
    "estimatedDefensiveRating" FLOAT,
    "netRating" FLOAT,
    "estimatedNetRating" FLOAT,

    "assistPercentage" FLOAT,
    "assistToTurnover" FLOAT,
    "assistRatio" FLOAT,
    "turnoverRatio" FLOAT,

    "offensiveReboundPercentage" FLOAT,
    "defensiveReboundPercentage" FLOAT,
    "reboundPercentage" FLOAT,

    "effectiveFieldGoalPercentage" FLOAT,
    "trueShootingPercentage" FLOAT,

    "usagePercentage" FLOAT,
    "estimatedUsagePercentage" FLOAT,
    "pace" FLOAT,
    "estimatedPace" FLOAT,
    "pacePer40" FLOAT,
    "possessions" FLOAT,

    "PIE" FLOAT,

    "percentageFieldGoalsAttempted2pt" FLOAT,
    "percentageFieldGoalsAttempted3pt" FLOAT,

    "percentagePoints2pt" FLOAT,
    "percentagePointsMidrange2pt" FLOAT,
    "percentagePoints3pt" FLOAT,
    "percentagePointsFastBreak" FLOAT,
    "percentagePointsFreeThrow" FLOAT,
    "percentagePointsOffTurnovers" FLOAT,
    "percentagePointsPaint" FLOAT,

    "freeThrowAttemptRate" FLOAT,

    "percentageAssisted2pt" FLOAT,
    "percentageUnassisted2pt" FLOAT,
    "percentageAssisted3pt" FLOAT,
    "percentageUnassisted3pt" FLOAT,
    "percentageAssistedFGM" FLOAT,
    "percentageUnassistedFGM" FLOAT,

    "pointsOffTurnovers" INTEGER,
    "pointsSecondChance" INTEGER,
    "pointsFastBreak" INTEGER,
    "pointsPaint" INTEGER,
    "oppPointsOffTurnovers" INTEGER,
    "oppPointsSecondChance" INTEGER,
    "oppPointsFastBreak" INTEGER,
    "oppPointsPaint" INTEGER,

    "oppEffectiveFieldGoalPercentage" FLOAT,
    "oppFreeThrowAttemptRate" FLOAT,
    "oppTeamTurnoverPercentage" FLOAT,
    "oppOffensiveReboundPercentage" FLOAT,

    "teamTurnoverPercentage" FLOAT,

    "percentagePersonalFouls" FLOAT,
    "percentagePersonalFoulsDrawn" FLOAT,

    PRIMARY KEY ("game_id", "team_id", "player_id")
);
"""

GAME_ODDS_SPREAD_SCHEMA = """
DROP TABLE IF EXISTS "game_spreads";
CREATE TABLE IF NOT EXISTS "game_spreads" (
    "team_id" TEXT NOT NULL,
    "dint" INTEGER NOT NULL,
    "bookmaker" TEXT,
    "last_update" TIMESTAMP,
    "price" FLOAT,
    "point" FLOAT,
    
    PRIMARY KEY ("team_id", "dint", "bookmaker", "last_update")
);
"""

GAME_ODDS_ML_SCHEMA = """
DROP TABLE IF EXISTS "game_ml";
CREATE TABLE IF NOT EXISTS "game_ml" (
    "team_id" TEXT NOT NULL,
    "dint" INTEGER NOT NULL,
    "bookmaker" TEXT,
    "last_update" TIMESTAMP,
    "price" FLOAT,

    PRIMARY KEY ("team_id", "dint", "bookmaker", "last_update")
);
"""

GAME_STATS_SCHEMA = """
DROP TABLE IF EXISTS "game_stats";
CREATE TABLE IF NOT EXISTS "game_stats" (
    "game_id" TEXT NOT NULL,
    "team_id" INTEGER NOT NULL,
    "is_home" BOOLEAN NOT NULL,
    "stat_type" TEXT NOT NULL,

    "minutes" FLOAT,

    "pointsOffTurnovers" FLOAT,
    "pointsSecondChance" FLOAT,
    "pointsFastBreak" FLOAT,
    "pointsPaint" FLOAT,

    "oppPointsOffTurnovers" FLOAT,
    "oppPointsSecondChance" FLOAT,
    "oppPointsFastBreak" FLOAT,
    "oppPointsPaint" FLOAT,

    "fieldGoalsMade" INTEGER,
    "fieldGoalsAttempted" INTEGER,
    "fieldGoalsPercentage" FLOAT,
    "threePointersMade" INTEGER,
    "threePointersAttempted" INTEGER,
    "threePointersPercentage" FLOAT,
    "freeThrowsMade" INTEGER,
    "freeThrowsAttempted" INTEGER,
    "freeThrowsPercentage" FLOAT,

    "reboundsOffensive" INTEGER,
    "reboundsDefensive" INTEGER,
    "reboundsTotal" INTEGER,

    "assists" INTEGER,
    "steals" INTEGER,
    "blocks" INTEGER,
    "blocksAgainst" FLOAT,
    "turnovers" INTEGER,
    "foulsPersonal" INTEGER,
    "foulsDrawn" INTEGER,

    "points" INTEGER,
    "plusMinusPoints" FLOAT,

    "effectiveFieldGoalPercentage" FLOAT,
    "trueShootingPercentage" FLOAT,
    "usagePercentage" FLOAT,
    "estimatedUsagePercentage" FLOAT,
    "offensiveRating" FLOAT,
    "estimatedOffensiveRating" FLOAT,
    "defensiveRating" FLOAT,
    "estimatedDefensiveRating" FLOAT,
    "netRating" FLOAT,
    "estimatedNetRating" FLOAT,
    "assistPercentage" FLOAT,
    "assistToTurnover" FLOAT,
    "assistRatio" FLOAT,
    "turnoverRatio" FLOAT,
    "teamTurnoverPercentage" FLOAT,
    "offensiveReboundPercentage" FLOAT,
    "defensiveReboundPercentage" FLOAT,
    "reboundPercentage" FLOAT,
    "freeThrowAttemptRate" FLOAT,

    "pace" FLOAT,
    "estimatedPace" FLOAT,
    "pacePer40" FLOAT,
    "possessions" FLOAT,
    "PIE" FLOAT,

    "oppEffectiveFieldGoalPercentage" FLOAT,
    "oppFreeThrowAttemptRate" FLOAT,
    "oppTeamTurnoverPercentage" FLOAT,
    "oppOffensiveReboundPercentage" FLOAT,

    "percentageFieldGoalsAttempted2pt" FLOAT,
    "percentageFieldGoalsAttempted3pt" FLOAT,
    "percentagePoints2pt" FLOAT,
    "percentagePointsMidrange2pt" FLOAT,
    "percentagePoints3pt" FLOAT,
    "percentagePointsFastBreak" FLOAT,
    "percentagePointsFreeThrow" FLOAT,
    "percentagePointsOffTurnovers" FLOAT,
    "percentagePointsPaint" FLOAT,

    "percentageAssisted2pt" FLOAT,
    "percentageUnassisted2pt" FLOAT,
    "percentageAssisted3pt" FLOAT,
    "percentageUnassisted3pt" FLOAT,
    "percentageAssistedFGM" FLOAT,
    "percentageUnassistedFGM" FLOAT,

    PRIMARY KEY ("game_id", "team_id", "stat_type")
);
"""