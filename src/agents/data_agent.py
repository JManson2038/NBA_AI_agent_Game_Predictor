import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    LeagueGameLog,
    TeamDashboardByGeneralSplits,
)
import config

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Fix: Browser-like headers to prevent nba_api timeouts ──
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}


def fetch_with_retry(endpoint_fn, retries=3, delay=5):
    # Retry wrapper for flaky nba_api calls
    for attempt in range(retries):
        try:
            return endpoint_fn()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e


class DataAgent:
    # Fetches, cleans, and engineers features from NBA game data

    def __init__(self, seasons=None):
        self.seasons = seasons or config.TRAIN_SEASONS
        self._init_team_mappings()

    # Team ID ↔ Abbreviation mappings
    def _init_team_mappings(self):
        all_teams = nba_teams.get_teams()
        config.TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in all_teams}
        config.TEAM_ABBR_TO_ID = {t["abbreviation"]: t["id"] for t in all_teams}
        self.teams = all_teams

    # Fetch game logs for all seasons
    def fetch_game_logs(self, force=False) -> pd.DataFrame:
        cache_path = CACHE_DIR / "game_logs.parquet"
        if cache_path.exists() and not force:
            print("[DataAgent] Loading cached game logs...")
            return pd.read_parquet(cache_path)

        print("[DataAgent] Fetching game logs from nba_api...")
        frames = []
        for season in tqdm(self.seasons, desc="Seasons"):
            try:
                # ── Fix: pass headers and timeout into every endpoint call ──
                log = fetch_with_retry(lambda s=season: LeagueGameLog(
                    season=s,
                    season_type_all_star="Regular Season",
                    player_or_team_abbreviation="T",
                    headers=HEADERS,
                    timeout=60,
                ))
                df = log.get_data_frames()[0]
                df["SEASON"] = season
                frames.append(df)
                time.sleep(0.6)  # Rate-limit respect
            except Exception as e:
                print(f" Error fetching {season}: {e}")
                continue

        game_logs = pd.concat(frames, ignore_index=True)
        game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"])
        game_logs.sort_values(["GAME_DATE", "GAME_ID"], inplace=True)
        game_logs.reset_index(drop=True, inplace=True)
        game_logs.to_parquet(cache_path)
        print(f"[DataAgent] Cached {len(game_logs)} team-game rows.")
        return game_logs

    # Build per-game matchup rows
    def build_matchup_dataset(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        cache_path = CACHE_DIR / "matchup_dataset.parquet"
        if cache_path.exists():
            print("[DataAgent] Loading cached matchup dataset...")
            return pd.read_parquet(cache_path)

        print("[DataAgent] Building matchup dataset...")
        gl = game_logs.copy()

        gl["IS_HOME"] = gl["MATCHUP"].str.contains("vs.").astype(int)
        gl["TEAM_ABBR"] = gl["TEAM_ABBREVIATION"]

        stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV",
                     "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]

        gl.sort_values(["TEAM_ID", "GAME_DATE"], inplace=True)

        for col in tqdm(stat_cols, desc="Rolling stats"):
            for w in [config.ROLLING_WINDOW_SHORT, config.ROLLING_WINDOW_LONG]:
                col_name = f"{col}_ROLL_{w}"
                gl[col_name] = (
                    gl.groupby("TEAM_ID")[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )

        home = gl[gl["IS_HOME"] == 1].copy()
        away = gl[gl["IS_HOME"] == 0].copy()

        home_cols = {c: f"HOME_{c}" for c in home.columns if c not in ["GAME_ID", "GAME_DATE", "SEASON"]}
        away_cols = {c: f"AWAY_{c}" for c in away.columns if c not in ["GAME_ID", "GAME_DATE", "SEASON"]}

        home = home.rename(columns=home_cols)
        away = away.rename(columns=away_cols)

        merged = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON"], how="inner")
        merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)

        for side in ["HOME", "AWAY"]:
            team_col = f"{side}_TEAM_ID"
            merged.sort_values(["GAME_DATE"], inplace=True)
            merged[f"{side}_REST_DAYS"] = (
                merged.groupby(team_col)["GAME_DATE"]
                .diff().dt.days.fillna(3).clip(upper=7)
            )

        merged.sort_values("GAME_DATE", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        merged.to_parquet(cache_path)
        print(f"[DataAgent] Built {len(merged)} matchup rows.")
        return merged

    # Select feature columns for modeling
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        rolling_cols = [c for c in df.columns if "ROLL_" in c]
        rest_cols = ["HOME_REST_DAYS", "AWAY_REST_DAYS"]
        injury_cols = [c for c in ["HOME_BEST_OUT_SCORE", "AWAY_BEST_OUT_SCORE", 
                       "BEST_OUT_SCORE_DIFF", "HOME_BEST_OUT_TIER", 
                       "AWAY_BEST_OUT_TIER"] if c in df.columns]
        return rolling_cols + rest_cols + injury_cols

    def run(self, force=False) -> pd.DataFrame:
        game_logs = self.fetch_game_logs(force=force)
        dataset = self.build_matchup_dataset(game_logs)
        feature_cols = self.get_feature_columns(dataset)
        print(f"[DataAgent] {len(feature_cols)} features ready.")
        return dataset


if __name__ == "__main__":
    agent = DataAgent()
    df = agent.run()
    print(df[["GAME_DATE", "HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "HOME_WIN"]].tail(10))