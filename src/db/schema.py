TEAMS_SCHEMA = """
DROP TABLE IF EXISTS teams;
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER,
    team_name TEXT,
    team_slug TEXT,
    PRIMARY KEY (team_id)
);"""

PLAYERS_SCHEMA = """
DROP TABLE IF EXISTS players;
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER,
    player_name TEXT,
    player_slug TEXT,
    PRIMARY KEY (player_id)
);"""

GAMES_META_SCHEMA = """
DROP TABLE IF EXISTS games;
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT,
    season TEXT,
    season_type TEXT,
    season_type_code TEXT,
    dint INTEGER,
    date DATETIME,
    home_team INTEGER,
    away_team INTEGER,
    PRIMARY KEY (game_id)
);"""

PLAYER_STATS_SCHEMA = """
DROP TABLE IF EXISTS player_stats;
CREATE TABLE IF NOT EXISTS player_stats (
    -- Identifiers
    player_id INTEGER,
    team_id INTEGER,
    game_id TEXT,
    position TEXT,

    -- Minutes Played
    minutes FLOAT,

    -- Box Score: Shooting
    fieldGoalsMade INTEGER,
    fieldGoalsAttempted INTEGER,
    fieldGoalsPercentage FLOAT,
    threePointersMade INTEGER,
    threePointersAttempted INTEGER,
    threePointersPercentage FLOAT,
    freeThrowsMade INTEGER,
    freeThrowsAttempted INTEGER,
    freeThrowsPercentage FLOAT,

    -- Box Score: Rebounding
    reboundsOffensive INTEGER,
    reboundsDefensive INTEGER,
    reboundsTotal INTEGER,

    -- Box Score: Playmaking / Defense
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    blocksAgainst INTEGER,
    turnovers INTEGER,
    foulsPersonal INTEGER,
    foulsDrawn INTEGER,

    -- Box Score: Scoring
    points INTEGER,
    plusMinusPoints FLOAT,

    -- Advanced Ratings
    offensiveRating FLOAT,
    estimatedOffensiveRating FLOAT,
    defensiveRating FLOAT,
    estimatedDefensiveRating FLOAT,
    netRating FLOAT,
    estimatedNetRating FLOAT,

    -- Assist / Turnover Efficiency
    assistPercentage FLOAT,
    assistToTurnover FLOAT,
    assistRatio FLOAT,
    turnoverRatio FLOAT,

    -- Rebound Efficiency
    offensiveReboundPercentage FLOAT,
    defensiveReboundPercentage FLOAT,
    reboundPercentage FLOAT,

    -- Shooting Efficiency
    effectiveFieldGoalPercentage FLOAT,
    trueShootingPercentage FLOAT,

    -- Usage and Pace
    usagePercentage FLOAT,
    estimatedUsagePercentage FLOAT,
    pace FLOAT,
    estimatedPace FLOAT,
    pacePer40 FLOAT,
    possessions FLOAT,

    -- Player Impact
    PIE FLOAT,

    -- Scoring Breakdown: Shot Selection (Attempts)
    percentageFieldGoalsAttempted2pt FLOAT,
    percentageFieldGoalsAttempted3pt FLOAT,
    
    -- Scoring Breakdown: Point Contribution
    percentagePoints2pt FLOAT,
    percentagePointsMidrange2pt FLOAT,
    percentagePoints3pt FLOAT,
    percentagePointsFastBreak FLOAT,
    percentagePointsFreeThrow FLOAT,
    percentagePointsOffTurnovers FLOAT,
    percentagePointsPaint FLOAT,
    
    -- Scoring Breakdown: FT Aggressiveness
    freeThrowAttemptRate FLOAT,

    -- Assist Types
    percentageAssisted2pt FLOAT,
    percentageUnassisted2pt FLOAT,
    percentageAssisted3pt FLOAT,
    percentageUnassisted3pt FLOAT,
    percentageAssistedFGM FLOAT,
    percentageUnassistedFGM FLOAT,

    -- Point Sources
    pointsOffTurnovers INTEGER,
    pointsSecondChance INTEGER,
    pointsFastBreak INTEGER,
    pointsPaint INTEGER,
    oppPointsOffTurnovers INTEGER,
    oppPointsSecondChance INTEGER,
    oppPointsFastBreak INTEGER,
    oppPointsPaint INTEGER,

    -- Opponent Defense Metrics
    oppEffectiveFieldGoalPercentage FLOAT,
    oppFreeThrowAttemptRate FLOAT,
    oppTeamTurnoverPercentage FLOAT,
    oppOffensiveReboundPercentage FLOAT,

    -- Team-level Contribution Metrics
    teamTurnoverPercentage FLOAT,

    -- Fouls/Impact
    percentagePersonalFouls FLOAT,
    percentagePersonalFoulsDrawn FLOAT,

    -- Primary Key
    PRIMARY KEY (game_id, player_id)
);
"""

GAME_STATS_SCHEMA = """
DROP TABLE IF EXISTS game_stats;
CREATE TABLE IF NOT EXISTS game_stats (
    -- Identifiers
    game_id TEXT,
    team_id INTEGER,
    is_home BOOLEAN,       -- 1 if home, 0 if away
    stat_type TEXT,        -- 'starters', 'bench', or 'statistics'

    -- Minutes (team total, e.g., 240:00)
    minutes FLOAT,

    -- Shooting
    fieldGoalsMade INTEGER,
    fieldGoalsAttempted INTEGER,
    fieldGoalsPercentage FLOAT,
    threePointersMade INTEGER,
    threePointersAttempted INTEGER,
    threePointersPercentage FLOAT,
    freeThrowsMade INTEGER,
    freeThrowsAttempted INTEGER,
    freeThrowsPercentage FLOAT,

    -- Rebounding
    reboundsOffensive INTEGER,
    reboundsDefensive INTEGER,
    reboundsTotal INTEGER,

    -- Playmaking / Defense
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    foulsPersonal INTEGER,
    foulsDrawn INTEGER,

    -- Scoring
    points INTEGER,
    plusMinusPoints FLOAT,

    -- Advanced Efficiency
    effectiveFieldGoalPercentage FLOAT,
    trueShootingPercentage FLOAT,
    usagePercentage FLOAT,
    estimatedUsagePercentage FLOAT,
    offensiveRating FLOAT,
    estimatedOffensiveRating FLOAT,
    defensiveRating FLOAT,
    estimatedDefensiveRating FLOAT,
    netRating FLOAT,
    estimatedNetRating FLOAT,
    assistPercentage FLOAT,
    assistToTurnover FLOAT,
    assistRatio FLOAT,
    turnoverRatio FLOAT,
    offensiveReboundPercentage FLOAT,
    defensiveReboundPercentage FLOAT,
    reboundPercentage FLOAT,
    freeThrowAttemptRate FLOAT,

    -- Possession / Pace
    pace FLOAT,
    estimatedPace FLOAT,
    pacePer40 FLOAT,
    possessions FLOAT,
    PIE FLOAT,

    -- Opponent Defense
    oppEffectiveFieldGoalPercentage FLOAT,
    oppFreeThrowAttemptRate FLOAT,
    oppTeamTurnoverPercentage FLOAT,
    oppOffensiveReboundPercentage FLOAT
);
"""