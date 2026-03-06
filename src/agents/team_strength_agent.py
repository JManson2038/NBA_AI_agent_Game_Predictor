import pandas as pd
import numpy as np
from collections import defaultdict

import config


class TeamStrengthAgent:

    def __init__(self):
        self.elo_ratings = defaultdict(lambda: config.ELO_INITIAL)
        self.elo_history = []  # Track ratings over time
        self.k = config.ELO_K_FACTOR
        self.home_adv = config.ELO_HOME_ADVANTAGE

    # Core Elo math
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        #Expected win probability for team A
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _margin_of_victory_mult(self, mov: float, elo_diff: float) -> float:
        return np.log(max(abs(mov), 1) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def _update_elo(self, winner_id, loser_id, mov: float, home_id=None):
        #Update Elo ratings after a single game
        w_elo = self.elo_ratings[winner_id]
        l_elo = self.elo_ratings[loser_id]

        # Add home-court advantage to expected score calculation
        w_adj = w_elo + (self.home_adv if winner_id == home_id else 0)
        l_adj = l_elo + (self.home_adv if loser_id == home_id else 0)

        w_expected = self._expected_score(w_adj, l_adj)
        elo_diff = w_adj - l_adj
        mov_mult = self._margin_of_victory_mult(mov, elo_diff)

        shift = self.k * mov_mult * (1 - w_expected)
        self.elo_ratings[winner_id] += shift
        self.elo_ratings[loser_id] -= shift

    # Season reversion (regress to mean between seasons)
    def _revert_to_mean(self):
        for team_id in self.elo_ratings:
            self.elo_ratings[team_id] = (
                config.ELO_SEASON_REVERT * self.elo_ratings[team_id]
                + (1 - config.ELO_SEASON_REVERT) * config.ELO_INITIAL
            )

    # Process full dataset and attach Elo to each game
    def run(self, matchup_df: pd.DataFrame) -> pd.DataFrame:
        
        print("[TeamStrengthAgent] Computing Elo ratings...")
        df = matchup_df.copy()
        df.sort_values("GAME_DATE", inplace=True)

        home_elos, away_elos, elo_diffs, elo_probs = [], [], [], []
        prev_season = None

        for _, row in df.iterrows():
            season = row["SEASON"]
            home_id = row["HOME_TEAM_ID"]
            away_id = row["AWAY_TEAM_ID"]

            # Revert Elo at the start of each new season
            if prev_season is not None and season != prev_season:
                self._revert_to_mean()
            prev_season = season

            # Capture PRE-GAME Elo (no leakage)
            h_elo = self.elo_ratings[home_id]
            a_elo = self.elo_ratings[away_id]
            home_elos.append(h_elo)
            away_elos.append(a_elo)

            # Expected win prob for home team (with home advantage)
            prob = self._expected_score(h_elo + self.home_adv, a_elo)
            elo_probs.append(prob)
            elo_diffs.append(h_elo - a_elo)

            # Update Elo after game
            home_pts = row.get("HOME_PTS", 0)
            away_pts = row.get("AWAY_PTS", 0)
            mov = abs(home_pts - away_pts)

            if row["HOME_WIN"] == 1:
                self._update_elo(home_id, away_id, mov, home_id=home_id)
            else:
                self._update_elo(away_id, home_id, mov, home_id=home_id)

        df["HOME_ELO"] = home_elos
        df["AWAY_ELO"] = away_elos
        df["ELO_DIFF"] = elo_diffs
        df["ELO_WIN_PROB"] = elo_probs

        print(f"[TeamStrengthAgent] Elo computed for {len(df)} games.")
        return df

    
    # Get current ratings snapshot
    def get_current_ratings(self) -> dict:
        abbr_ratings = {}
        for team_id, elo in self.elo_ratings.items():
            abbr = config.TEAM_ID_TO_ABBR.get(team_id, str(team_id))
            abbr_ratings[abbr] = round(elo, 1)
        return dict(sorted(abbr_ratings.items(), key=lambda x: -x[1]))

    def get_elo(self, team_abbr: str) -> float:
        team_id = config.TEAM_ABBR_TO_ID.get(team_abbr)
        if team_id is None:
            raise ValueError(f"Unknown team: {team_abbr}")
        return self.elo_ratings[team_id]