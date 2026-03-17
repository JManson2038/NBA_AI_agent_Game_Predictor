import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
import config

CACHE_DIR = Path("cache")

HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://stats.nba.com/',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
}


def fetch_player_game_logs():
    """Fetch player-level game logs for all training seasons."""
    cache_path = CACHE_DIR / "player_game_logs.parquet"
    if cache_path.exists():
        print("Loading cached player game logs...")
        return pd.read_parquet(cache_path)

    print("Fetching player game logs...")
    frames = []
    for season in config.TRAIN_SEASONS:
        print(f"  {season}...")
        try:
            log = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                headers=HEADERS,
                timeout=120,
            )
            df = log.get_data_frames()[0]
            df["SEASON"] = season
            frames.append(df)
            time.sleep(3)  # be gentle with the API
        except Exception as e:
            print(f"  Error: {e}")
            continue

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(cache_path)
    print(f"Cached {len(result)} player-game rows.")
    return result

def detect_missing_players(player_logs):
    """For each game, find key players who didn't play."""
    print("Detecting missing players per game...")

    # Step 1: Calculate season averages per player
    season_avgs = (
        player_logs.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"])
        .agg(GP=("MIN", "count"), AVG_MIN=("MIN", "mean"), AVG_PTS=("PTS", "mean"))
        .reset_index()
    )

    # Only care about players averaging 20+ MPG
    regulars = season_avgs[season_avgs["AVG_MIN"] >= 20].copy()
    print(f"  {len(regulars)} regular players (20+ MPG) across all seasons")

    # Step 2: For each game, check which regulars are missing
    # Get set of (GAME_ID, TEAM_ID, PLAYER_ID) that actually played
    played = set(
        zip(player_logs["GAME_ID"], player_logs["TEAM_ID"], player_logs["PLAYER_ID"])
    )

    # Get all unique (GAME_ID, TEAM_ID) combos
    games = player_logs[["GAME_ID", "TEAM_ID", "SEASON"]].drop_duplicates()

    missing_records = []
    for _, game in games.iterrows():
        gid = game["GAME_ID"]
        tid = game["TEAM_ID"]
        season = game["SEASON"]

        # Get regulars for this team this season
        team_regulars = regulars[
            (regulars["SEASON"] == season) & (regulars["TEAM_ID"] == tid)
        ]

        for _, player in team_regulars.iterrows():
            if (gid, tid, player["PLAYER_ID"]) not in played:
                missing_records.append({
                    "GAME_ID":           gid,
                    "TEAM_ID":           tid,
                    "TEAM_ABBREVIATION": player["TEAM_ABBREVIATION"],
                    "PLAYER_ID":         player["PLAYER_ID"],
                    "PLAYER_NAME":       player["PLAYER_NAME"],
                    "AVG_MIN":           round(player["AVG_MIN"], 1),
                    "AVG_PTS":           round(player["AVG_PTS"], 1),
                    "SEASON":            season,
                })

    missing_df = pd.DataFrame(missing_records)
    print(f"  Found {len(missing_df)} missing-player instances")
    return missing_df

def score_missing_players(missing_df, player_logs):
    """Score each missing player using the same formula as PlayerValueNN."""
    from src.models.player_value_nn import generate_importance_label, score_to_tier

    print("Scoring missing players...")

    # Build season averages with full stats for scoring
    season_avgs = (
        player_logs.groupby(["SEASON", "PLAYER_ID", "TEAM_ID"])
        .agg(
            MIN=("MIN", "mean"),
            PTS=("PTS", "mean"),
            REB=("REB", "mean"),
            AST=("AST", "mean"),
            STL=("STL", "mean"),
            BLK=("BLK", "mean"),
            TOV=("TOV", "mean"),
            FG_PCT=("FG_PCT", "mean"),
            GP=("MIN", "count"),
        )
        .reset_index()
    )

    # Score each player-season
    scores = {}
    for _, row in season_avgs.iterrows():
        record = row.to_dict()
        score = generate_importance_label(record)
        scores[(row["SEASON"], row["PLAYER_ID"], row["TEAM_ID"])] = score

    # Attach scores to missing players
    missing_df["SCORE"] = missing_df.apply(
        lambda r: scores.get((r["SEASON"], r["PLAYER_ID"], r["TEAM_ID"]), 0),
        axis=1
    )
    missing_df["TIER"] = missing_df["SCORE"].apply(score_to_tier)

    print(f"  Scored {len(missing_df)} missing-player instances")
    print(f"  Score range: {missing_df['SCORE'].min():.3f} - {missing_df['SCORE'].max():.3f}")
    return missing_df


def score_missing_players(missing_df, player_logs):
    """Score each missing player using the same formula as PlayerValueNN."""
    from src.models.player_value_nn import generate_importance_label, score_to_tier

    print("Scoring missing players...")

    # Build season averages with full stats for scoring
    season_avgs = (
        player_logs.groupby(["SEASON", "PLAYER_ID", "TEAM_ID"])
        .agg(
            MIN=("MIN", "mean"),
            PTS=("PTS", "mean"),
            REB=("REB", "mean"),
            AST=("AST", "mean"),
            STL=("STL", "mean"),
            BLK=("BLK", "mean"),
            TOV=("TOV", "mean"),
            FG_PCT=("FG_PCT", "mean"),
            GP=("MIN", "count"),
        )
        .reset_index()
    )

    # Score each player-season
    scores = {}
    for _, row in season_avgs.iterrows():
        record = row.to_dict()
        score = generate_importance_label(record)
        scores[(row["SEASON"], row["PLAYER_ID"], row["TEAM_ID"])] = score

    # Attach scores to missing players
    missing_df["SCORE"] = missing_df.apply(
        lambda r: scores.get((r["SEASON"], r["PLAYER_ID"], r["TEAM_ID"]), 0),
        axis=1
    )
    missing_df["TIER"] = missing_df["SCORE"].apply(score_to_tier)

    print(f"  Scored {len(missing_df)} missing-player instances")
    print(f"  Score range: {missing_df['SCORE'].min():.3f} - {missing_df['SCORE'].max():.3f}")
    return missing_df


def build_best_out_features(missing_df):
    """Per game+team, find the best missing player. Then pivot to per-game features."""
    print("Building best-out features per game...")

    # Best missing player per team per game
    best = missing_df.loc[missing_df.groupby(["GAME_ID", "TEAM_ID"])["SCORE"].idxmax()]

    # We need home/away split — use MATCHUP to detect
    # But easier: merge with matchup_dataset which already has home/away team IDs
    matchup = pd.read_parquet(CACHE_DIR / "matchup_dataset.parquet")

    # Build lookup: GAME_ID + TEAM_ID -> best out score
    best_lookup = best.set_index(["GAME_ID", "TEAM_ID"])["SCORE"].to_dict()

    tier_map = {"Franchise": 4, "Star": 3, "Key Rotation": 2, "Bench": 1, "Two-Way": 0.5, "None": 0}
    best_tier_lookup = best.set_index(["GAME_ID", "TEAM_ID"])["TIER"].to_dict()

    matchup["HOME_BEST_OUT_SCORE"] = matchup.apply(
        lambda r: best_lookup.get((r["GAME_ID"], r["HOME_TEAM_ID"]), 0.0), axis=1
    )
    matchup["AWAY_BEST_OUT_SCORE"] = matchup.apply(
        lambda r: best_lookup.get((r["GAME_ID"], r["AWAY_TEAM_ID"]), 0.0), axis=1
    )
    matchup["BEST_OUT_SCORE_DIFF"] = matchup["HOME_BEST_OUT_SCORE"] - matchup["AWAY_BEST_OUT_SCORE"]

    matchup["HOME_BEST_OUT_TIER"] = matchup.apply(
        lambda r: tier_map.get(best_tier_lookup.get((r["GAME_ID"], r["HOME_TEAM_ID"]), "None"), 0), axis=1
    )
    matchup["AWAY_BEST_OUT_TIER"] = matchup.apply(
        lambda r: tier_map.get(best_tier_lookup.get((r["GAME_ID"], r["AWAY_TEAM_ID"]), "None"), 0), axis=1
    )

    # Save updated dataset
    out_path = CACHE_DIR / "matchup_dataset.parquet"
    matchup.to_parquet(out_path)
    print(f"  Saved updated matchup dataset with injury features")
    print(f"  Non-zero HOME_BEST_OUT_SCORE: {(matchup['HOME_BEST_OUT_SCORE'] > 0).sum()} / {len(matchup)}")
    return matchup

if __name__ == "__main__":
    df = fetch_player_game_logs()
    missing = detect_missing_players(df)
    missing = score_missing_players(missing, df)
    matchup = build_best_out_features(missing)