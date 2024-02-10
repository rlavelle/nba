TEAMS_SCHEMA = """
DROP TABLE IF EXISTS teams;
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT,
    team_slug TEXT
);"""

PLAYERS_SCHEMA = """
DROP TABLE IF EXISTS players;
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT,
    player_slug TEXT
);"""

GAMES_SCHEMA = """
DROP TABLE IF EXISTS games;
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season TEXT,
    season_type TEXT,
    season_type_code TEXT,
    dint INTEGER,
    date DATETIME,
    home_team INTEGER,
    away_team INTEGER,
    home_score INTEGER,
    away_score INTEGER
);"""

STATS_SCHEMA = """
DROP TABLE IF EXISTS player_stats;
CREATE TABLE IF NOT EXISTS player_stats (
    player_id INTEGER,
    team_id INTEGER,
    game_id INTEGER,
    position TEXT,
    minutes FLOAT,
    fieldGoalsMade INTEGER,
    fieldGoalsAttempted INTEGER,
    fieldGoalsPercentage FLOAT,
    threePointersMade INTEGER,
    threePointersAttempted INTEGER,
    threePointersPercentage FLOAT,
    freeThrowsMade INTEGER,
    freeThrowsAttempted INTEGER,
    freeThrowsPercentage FLOAT,
    reboundsOffensive INTEGER,
    reboundsDefensive INTEGER,
    reboundsTotal INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    foulsPersonal INTEGER,
    points INTEGER,
    plusMinusPoints FLOAT,
    PRIMARY KEY (game_id, player_id)
);"""