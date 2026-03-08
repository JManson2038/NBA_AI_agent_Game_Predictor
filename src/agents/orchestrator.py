import numpy as np
import pandas as pd
from src.agents.team_strength_agent import TeamStrengthAgent
from src.agents.matchup_agent import MatchupAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.confidence_agent import ConfidenceAgent
from src.agents.injury_agent import InjuryAgent
import config


class Orchestrator:
    def __init__(self, strength_agent=None):
        self.strength_agent = strength_agent or TeamStrengthAgent()
        self.matchup_agent = MatchupAgent()
        self.prediction_agent = PredictionAgent()
        self.confidence_agent = ConfidenceAgent()
        self.injury_agent = InjuryAgent()        

    def predict_from_row(self, row, feature_cols):
       # Generate a full prediction from a dataset row
        X = np.array([row[c] for c in feature_cols]).reshape(1, -1)
        pred = self.prediction_agent.predict(X)

        game_features = {"ELO_DIFF": row.get("ELO_DIFF", 0),
                         "HOME_MOMENTUM": row.get("HOME_MOMENTUM", 0),
                         "AWAY_MOMENTUM": row.get("AWAY_MOMENTUM", 0)}
        for c in feature_cols:
            game_features[c] = row.get(c, 0)

        conf = self.confidence_agent.score(pred, game_features)
        cal_prob = conf["calibrated_probability"]
        raw_prob = conf["raw_probability"]

        home_abbr = row.get("HOME_TEAM_ABBR", "HOME")
        away_abbr = row.get("AWAY_TEAM_ABBR", "AWAY")

        if cal_prob >= 0.5:
            winner = home_abbr
            win_prob = cal_prob
        else:
            winner = away_abbr
            win_prob = 1 - cal_prob

        row_with_gap = row.copy() if isinstance(row, pd.Series) else pd.Series(row)
        row_with_gap["MODEL_GAP"] = conf["breakdown"].get("model_gap", 0)
        key_factors = self.matchup_agent.explain_matchup(row_with_gap)

        return {
            "home": home_abbr,
            "away": away_abbr,
            "winner": winner,
            "win_probability": round(float(win_prob), 3),
            "home_win_prob": round(float(cal_prob), 3),
            "raw_home_win_prob": round(float(raw_prob), 3),
            "spread": round(float(pred["spread"]), 1),
            "confidence": conf["confidence_label"],
            "confidence_score": conf["confidence_score"],
            "confidence_breakdown": conf["breakdown"],
            "key_factors": key_factors,
            "model_detail": {
                "lr_prob": round(float(pred["lr_prob"]), 3),
                "xgb_prob": round(float(pred["xgb_prob"]), 3),
                "model_gap": conf["breakdown"].get("model_gap", 0),
            },
        }

    def predict_matchup(self, home_abbr, away_abbr, latest_features, feature_cols):
       # Predict a game given two team abbreviations, with injury adjustments
        home_abbr = home_abbr.upper()
        away_abbr = away_abbr.upper()

        home_rows = latest_features[latest_features["HOME_TEAM_ABBR"] == home_abbr]
        away_rows = latest_features[latest_features["AWAY_TEAM_ABBR"] == away_abbr]

        if home_rows.empty or away_rows.empty:
            all_home = latest_features[
                (latest_features["HOME_TEAM_ABBR"] == home_abbr) |
                (latest_features["AWAY_TEAM_ABBR"] == home_abbr)
            ]
            all_away = latest_features[
                (latest_features["HOME_TEAM_ABBR"] == away_abbr) |
                (latest_features["AWAY_TEAM_ABBR"] == away_abbr)
            ]
            if all_home.empty or all_away.empty:
                return {"error": f"Insufficient data for {home_abbr} vs {away_abbr}"}

        row = home_rows.iloc[-1].copy() if not home_rows.empty else all_home.iloc[-1].copy()

        if not away_rows.empty:
            away_row = away_rows.iloc[-1]
            for c in feature_cols:
                if c.startswith("AWAY_"):
                    row[c] = away_row.get(c, row.get(c, 0))

        # ── Base Elo ──
        base_home_elo = self.strength_agent.get_elo(home_abbr)
        base_away_elo = self.strength_agent.get_elo(away_abbr)

        # ── Injury-adjusted Elo ──
        adj_home_elo, home_impact, home_details = self.injury_agent.get_adjusted_elo(
            home_abbr, base_home_elo
        )
        adj_away_elo, away_impact, away_details = self.injury_agent.get_adjusted_elo(
            away_abbr, base_away_elo
        )

        row["HOME_ELO"] = adj_home_elo
        row["AWAY_ELO"] = adj_away_elo
        row["ELO_DIFF"] = adj_home_elo - adj_away_elo
        row["ELO_WIN_PROB"] = 1.0 / (
            1.0 + 10 ** (-(row["ELO_DIFF"] + config.ELO_HOME_ADVANTAGE) / 400)
        )
        row["HOME_TEAM_ABBR"] = home_abbr
        row["AWAY_TEAM_ABBR"] = away_abbr

        result = self.predict_from_row(row, feature_cols)

        # ── Attach injury context to output ──
        result["injury_report"] = {
            "home": {"team": home_abbr, "impact": home_impact, "players": home_details,
                     "base_elo": base_home_elo, "adjusted_elo": adj_home_elo},
            "away": {"team": away_abbr, "impact": away_impact, "players": away_details,
                     "base_elo": base_away_elo, "adjusted_elo": adj_away_elo},
        }

        return result

    def predict_batch(self, df, feature_cols):
        # Predict all games in a dataframe (no injury adjustment for batch)
        results = []
        for _, row in df.iterrows():
            try:
                pred = self.predict_from_row(row, feature_cols)
                pred["actual_home_win"] = int(row.get("HOME_WIN", -1))
                results.append(pred)
            except Exception:
                continue
        return results


def format_prediction(pred):
   # Pretty-print a single prediction including injury report
    if "error" in pred:
        return "Error: " + pred["error"]

    raw_home_p = pred.get("raw_home_win_prob", pred["home_win_prob"])
    cal_home_p = pred["home_win_prob"]
    raw_winner_p = raw_home_p if pred["winner"] == pred["home"] else 1 - raw_home_p
    cal_note = ""
    if abs(raw_winner_p - pred["win_probability"]) > 0.01:
        cal_note = f"  (raw: {round(raw_winner_p * 100, 1)}%)"

    gap = pred["model_detail"].get("model_gap", 0)
    conflict_note = f"  !! MODELS DISAGREE (gap={round(gap,2)})" if gap > 0.20 else ""

    lines = [
        "",
        "=" * 55,
        f"  {pred['away']}  @  {pred['home']}",
        "=" * 55,
        "",
        f"  Winner:       {pred['winner']}",
        f"  Win Prob:     {round(pred['win_probability']*100,1)}%{cal_note}",
        f"  Spread:       {pred['spread']}",
        f"  Confidence:   {pred['confidence']} ({pred['confidence_score']}){conflict_note}",
        "",
        "  Key Factors:",
    ]
    for factor in pred["key_factors"]:
        lines.append(f"    - {factor}")

    # ── Injury report section ──
    inj = pred.get("injury_report")
    if inj:
        lines.append("")
        lines.append("  INJURY IMPACT:")
        for side in ["home", "away"]:
            team_inj = inj[side]
            label = "HOME" if side == "home" else "AWAY"
            if team_inj["players"]:
                elo_diff = team_inj["adjusted_elo"] - team_inj["base_elo"]
                lines.append(
                    f"    {label} {team_inj['team']}: Elo {team_inj['base_elo']} → "
                    f"{team_inj['adjusted_elo']} ({round(elo_diff,1)})"
                )
                for p in team_inj["players"]:
                    lines.append(
                        f"      ✗ {p['player']:<22} {p['status']:<12} [{p['tier']}]"
                    )
            else:
                lines.append(f"    {label} {team_inj['team']}: No injuries")

    lines += [
        "",
        f"  Models:  LR={pred['model_detail']['lr_prob']}",
        f"           XGB={pred['model_detail']['xgb_prob']}",
        f"           Gap={round(gap, 3)}",
        "=" * 55,
    ]

    return "\n".join(lines)