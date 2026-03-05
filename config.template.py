
#  Seasons to fetch (nba_api uses "2024-25" format)
# Adjust range for more/less historical data
TRAIN_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25",
]
CURRENT_SEASON = "2025-26"


ELO_INITIAL = 1500
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 100
ELO_SEASON_REVERT = 0.75  # Revert 25% toward mean each new season

ROLLING_WINDOW_SHORT = 5    # Last 5 games
ROLLING_WINDOW_LONG = 15    # Last 15 games

# Feature Columns (engineered per-game for home & away) 
BASE_STATS = [
    "W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "STL", "BLK", "TOV", "PTS",
    "PLUS_MINUS",
]

ADVANCED_STATS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PACE", "EFG_PCT", "TS_PCT",
    "TM_TOV_PCT", "OREB_PCT", "DREB_PCT",
]

TEST_SEASON = "2024-25"  # Hold-out season for evaluation
RANDOM_STATE = 42

#Team Abbreviation Mapping (nba_api id to abbreviation)
TEAM_ID_TO_ABBR = {}
TEAM_ABBR_TO_ID = {}

# Confidence Thresholds 
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.55

ODDS_API_KEY = "86f7c12e4a44f943f8c737b20a40f6bc"