
import pandas as pd
import numpy as np
import config


class MatchupAgent:
   # Detects stylistic mismatches between home and away teams

    def __init__(self):
        self.matchup_signals = []
        #  Momentum damping factor (reduce hot streak weight)
        self.momentum_damping = 0.5  # 50% reduction

    def run(self, df):
        # matchup-derived features to the dataset
        print("[MatchupAgent] Computing matchup signals...")
        df = df.copy()
        short = config.ROLLING_WINDOW_SHORT
        long = config.ROLLING_WINDOW_LONG

        # Pace differential
        h_pts = "HOME_PTS_ROLL_" + str(short)
        a_pts = "AWAY_PTS_ROLL_" + str(short)
        if h_pts in df.columns and a_pts in df.columns:
            df["PACE_DIFF"] = df[h_pts] - df[a_pts]

        # 3PT shooting advantage
        h3 = "HOME_FG3_PCT_ROLL_" + str(short)
        a3 = "AWAY_FG3_PCT_ROLL_" + str(short)
        if h3 in df.columns and a3 in df.columns:
            df["THREE_PT_ADVANTAGE"] = df[h3] - df[a3]

        # Rebounding edge
        hr = "HOME_REB_ROLL_" + str(short)
        ar = "AWAY_REB_ROLL_" + str(short)
        if hr in df.columns and ar in df.columns:
            df["REBOUND_EDGE"] = df[hr] - df[ar]

        # Turnover differential (lower is better)
        ht = "HOME_TOV_ROLL_" + str(short)
        at = "AWAY_TOV_ROLL_" + str(short)
        if ht in df.columns and at in df.columns:
            df["TOV_DIFF"] = df[at] - df[ht]

        # Assist differential
        ha = "HOME_AST_ROLL_" + str(short)
        aa = "AWAY_AST_ROLL_" + str(short)
        if ha in df.columns and aa in df.columns:
            df["AST_DIFF"] = df[ha] - df[aa]

        # ── Fix #3: Damped momentum ──
        # Raw momentum (short-term vs long-term plus/minus)
        hpm_s = "HOME_PLUS_MINUS_ROLL_" + str(short)
        hpm_l = "HOME_PLUS_MINUS_ROLL_" + str(long)
        apm_s = "AWAY_PLUS_MINUS_ROLL_" + str(short)
        apm_l = "AWAY_PLUS_MINUS_ROLL_" + str(long)

        if all(c in df.columns for c in [hpm_s, hpm_l, apm_s, apm_l]):
            raw_home_mom = df[hpm_s] - df[hpm_l]
            raw_away_mom = df[apm_s] - df[apm_l]

            # Apply damping to prevent double-counting with Elo
            df["HOME_MOMENTUM"] = raw_home_mom * self.momentum_damping
            df["AWAY_MOMENTUM"] = raw_away_mom * self.momentum_damping

            # Fix #3 conditional: Only apply momentum if rest >= 2 days
            if "HOME_REST_DAYS" in df.columns:
                home_rested = (df["HOME_REST_DAYS"] >= 2).astype(float)
                df["HOME_MOMENTUM"] = df["HOME_MOMENTUM"] * home_rested
            if "AWAY_REST_DAYS" in df.columns:
                away_rested = (df["AWAY_REST_DAYS"] >= 2).astype(float)
                df["AWAY_MOMENTUM"] = df["AWAY_MOMENTUM"] * away_rested

            df["MOMENTUM_DIFF"] = df["HOME_MOMENTUM"] - df["AWAY_MOMENTUM"]

        # Plus/Minus differential
        if hpm_s in df.columns and apm_s in df.columns:
            df["PM_DIFF"] = df[hpm_s] - df[apm_s]

        # ── Fix #4: Context-aware rest advantage ──
        if "HOME_REST_DAYS" in df.columns and "AWAY_REST_DAYS" in df.columns:
            raw_rest_adv = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]

            # Scale rest by team pace (faster pace teams benefit less from rest)
            if h_pts in df.columns and a_pts in df.columns:
                # Higher scoring = faster pace = less rest benefit
                home_pace_factor = 1.0 - (df[h_pts] - 95).clip(0, 20) / 40
                away_pace_factor = 1.0 - (df[a_pts] - 95).clip(0, 20) / 40
                pace_scale = (home_pace_factor + away_pace_factor) / 2
                df["REST_ADVANTAGE"] = raw_rest_adv * pace_scale.clip(0.5, 1.0)
            else:
                df["REST_ADVANTAGE"] = raw_rest_adv

            # Keep raw rest for reference
            df["REST_ADVANTAGE_RAW"] = raw_rest_adv

        # FG% differential
        hfg = "HOME_FG_PCT_ROLL_" + str(short)
        afg = "AWAY_FG_PCT_ROLL_" + str(short)
        if hfg in df.columns and afg in df.columns:
            df["FG_PCT_DIFF"] = df[hfg] - df[afg]

        print("[MatchupAgent] Added matchup features.")
        return df

    def explain_matchup(self, row):
        # Generate key factor explanations for a single game
        factors = []

        elo_diff = row.get("ELO_DIFF", 0)
        if elo_diff > 50:
            factors.append("Home team Elo advantage (+" + str(round(elo_diff)) + ")")
        elif elo_diff < -50:
            factors.append("Away team Elo advantage (" + str(round(elo_diff)) + ")")

        rest_adv = row.get("REST_ADVANTAGE", 0)
        rest_raw = row.get("REST_ADVANTAGE_RAW", rest_adv)
        if abs(rest_raw) >= 2:
            if rest_adv >= 1:
                factors.append("Home rest advantage (pace-adjusted)")
            elif rest_adv <= -1:
                factors.append("Away rest advantage (pace-adjusted)")
            elif abs(rest_raw) >= 2:
                factors.append("Rest edge reduced by pace context")

        three_pt = row.get("THREE_PT_ADVANTAGE", 0)
        if three_pt > 0.03:
            factors.append("Home team shooting better from 3PT")
        elif three_pt < -0.03:
            factors.append("Away team shooting better from 3PT")

        mom_diff = row.get("MOMENTUM_DIFF", 0)
        if mom_diff > 1.5:  # Raised threshold due to damping
            factors.append("Home team trending up (damped)")
        elif mom_diff < -1.5:
            factors.append("Away team trending up (damped)")

        reb = row.get("REBOUND_EDGE", 0)
        if reb > 3:
            factors.append("Home team dominating the boards")
        elif reb < -3:
            factors.append("Away team dominating the boards")

        tov = row.get("TOV_DIFF", 0)
        if tov > 2:
            factors.append("Home team takes better care of the ball")
        elif tov < -2:
            factors.append("Away team takes better care of the ball")

        # Flag model conflict in explanation
        model_gap = row.get("MODEL_GAP", 0)
        if model_gap > 0.20:
            factors.append("WARNING: Models disagree significantly (gap=" + str(round(model_gap, 2)) + ")")

        if not factors:
            factors.append("Evenly matched game - no clear edges")

        return factors

    def get_matchup_feature_columns(self, df):
        # Return the matchup-specific feature columns 
        matchup_cols = [
            "PACE_DIFF", "THREE_PT_ADVANTAGE", "REBOUND_EDGE",
            "TOV_DIFF", "AST_DIFF", "HOME_MOMENTUM", "AWAY_MOMENTUM",
            "MOMENTUM_DIFF", "PM_DIFF", "REST_ADVANTAGE", "FG_PCT_DIFF",
        ]
        return [c for c in matchup_cols if c in df.columns]