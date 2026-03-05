"""
backtest_injury.py — Phase 1 & 2: Compare injury model versions

Tests three models over your historical dataset:
  Model A — No injury adjustment (pure Elo + features)
  Model B — Current fixed penalty model
  Model C — Phase 3 calibrated tiered model

Metrics per model:
  - Accuracy (win %)
  - Units won/lost at -110
  - ROI %
  - Average edge vs 50/50 baseline
  - Brier score
  - Accuracy by confidence tier

Usage:
    python backtest_injury.py                   # all models, 2024-25
    python backtest_injury.py --season 2023-24
    python backtest_injury.py --model C         # one model only
    python backtest_injury.py --min-edge 0.04   # only high-edge games
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss

from data_agent import DataAgent
from team_strength_agent import TeamStrengthAgent
from matchup_agent import MatchupAgent
from prediction_agent import PredictionAgent
from confidence_agent import ConfidenceAgent
import config

CACHE_DIR = Path("cache")


# ─────────────────────────────────────────────────────────────────
#  Betting math
# ─────────────────────────────────────────────────────────────────

def american_to_implied(odds=-110):
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def calculate_units(predictions, min_edge=0.0, odds=-110):
    """
    Calculate units won/lost betting 1 unit per game at given odds.
    Only bet when model edge > min_edge vs implied probability.
    """
    implied = american_to_implied(odds)
    payout = 100 / abs(odds)  # Profit per unit at -110 = 0.909

    units = 0.0
    bets = 0

    for p in predictions:
        prob = p["home_win_prob"]
        actual = p.get("actual_home_win", -1)
        if actual not in (0, 1):
            continue

        # Edge = model probability vs implied market probability
        edge = prob - implied if prob >= 0.5 else (1 - prob) - implied
        if edge < min_edge:
            continue

        bets += 1
        # Bet on whichever side model favors
        if prob >= 0.5:
            won = actual == 1
        else:
            won = actual == 0

        units += payout if won else -1.0

    return round(units, 2), bets


def roi(units, bets):
    if bets == 0:
        return 0.0
    return round((units / bets) * 100, 2)


# ─────────────────────────────────────────────────────────────────
#  Model runners
# ─────────────────────────────────────────────────────────────────

def load_pipeline(season):
    """Load all agents and dataset."""
    cache_path = CACHE_DIR / "matchup_dataset.parquet"
    if not cache_path.exists():
        print("No dataset found. Run: python train.py")
        return None, None, None, None

    dataset = pd.read_parquet(cache_path)

    data_agent = DataAgent()
    strength_agent = TeamStrengthAgent()
    dataset = strength_agent.run(dataset)

    matchup_agent = MatchupAgent()
    dataset = matchup_agent.run(dataset)

    rolling_cols = data_agent.get_feature_columns(dataset)
    elo_cols = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF", "ELO_WIN_PROB"]
    matchup_cols = matchup_agent.get_matchup_feature_columns(dataset)
    feature_cols = list(dict.fromkeys(rolling_cols + elo_cols + matchup_cols))

    prediction_agent = PredictionAgent()
    prediction_agent._load_models()

    test_df = dataset[dataset["SEASON"] == season].dropna(subset=feature_cols)
    print(f"  Loaded {len(test_df)} games from {season}")

    return test_df, feature_cols, strength_agent, prediction_agent


def run_model_a(test_df, feature_cols, prediction_agent, confidence_agent):
    """Model A — No injury adjustment."""
    results = []
    for _, row in test_df.iterrows():
        try:
            X = np.array([row[c] for c in feature_cols]).reshape(1, -1)
            pred = prediction_agent.predict(X)
            conf = confidence_agent.score(pred)
            cal_prob = conf["calibrated_probability"]

            results.append({
                "home_win_prob": cal_prob,
                "confidence": conf["confidence_label"],
                "actual_home_win": int(row.get("HOME_WIN", -1)),
            })
        except Exception:
            continue
    return results


def run_model_b(test_df, feature_cols, prediction_agent, confidence_agent, strength_agent):
    """Model B — Current fixed penalty model (flat -140 Elo)."""
    # Simulate old flat penalty: any injured player = -140 Elo regardless of tier
    FLAT_PENALTY = 140

    injury_file = Path("injuries.json")
    if not injury_file.exists():
        print("  No injuries.json found for Model B — running without injuries")
        return run_model_a(test_df, feature_cols, prediction_agent, confidence_agent)

    with open(injury_file) as f:
        injuries = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    results = []
    for _, row in test_df.iterrows():
        try:
            home = row.get("HOME_TEAM_ABBR", "")
            away = row.get("AWAY_TEAM_ABBR", "")

            # Flat penalty per injured player
            home_pen = sum(
                FLAT_PENALTY for s in injuries.get(home, {}).values()
                if s == "out"
            ) + sum(
                FLAT_PENALTY * 0.5 for s in injuries.get(home, {}).values()
                if s == "questionable"
            )
            away_pen = sum(
                FLAT_PENALTY for s in injuries.get(away, {}).values()
                if s == "out"
            ) + sum(
                FLAT_PENALTY * 0.5 for s in injuries.get(away, {}).values()
                if s == "questionable"
            )

            mod_row = row.copy()
            mod_row["HOME_ELO"] = row["HOME_ELO"] - home_pen
            mod_row["AWAY_ELO"] = row["AWAY_ELO"] - away_pen
            mod_row["ELO_DIFF"] = mod_row["HOME_ELO"] - mod_row["AWAY_ELO"]
            mod_row["ELO_WIN_PROB"] = 1.0 / (
                1.0 + 10 ** (-(mod_row["ELO_DIFF"] + config.ELO_HOME_ADVANTAGE) / 400)
            )

            X = np.array([mod_row[c] for c in feature_cols]).reshape(1, -1)
            pred = prediction_agent.predict(X)
            conf = confidence_agent.score(pred)
            cal_prob = conf["calibrated_probability"]

            results.append({
                "home_win_prob": cal_prob,
                "confidence": conf["confidence_label"],
                "actual_home_win": int(row.get("HOME_WIN", -1)),
            })
        except Exception:
            continue
    return results


def run_model_c(test_df, feature_cols, prediction_agent, confidence_agent):
    """Model C — Phase 3 calibrated tiered injury model."""
    from injury_agent import InjuryAgent
    injury_agent = InjuryAgent()

    results = []
    for _, row in test_df.iterrows():
        try:
            home = row.get("HOME_TEAM_ABBR", "")
            away = row.get("AWAY_TEAM_ABBR", "")

            home_pen, _ = injury_agent.compute_team_impact(home)
            away_pen, _ = injury_agent.compute_team_impact(away)

            mod_row = row.copy()
            mod_row["HOME_ELO"] = row["HOME_ELO"] - home_pen
            mod_row["AWAY_ELO"] = row["AWAY_ELO"] - away_pen
            mod_row["ELO_DIFF"] = mod_row["HOME_ELO"] - mod_row["AWAY_ELO"]
            mod_row["ELO_WIN_PROB"] = 1.0 / (
                1.0 + 10 ** (-(mod_row["ELO_DIFF"] + config.ELO_HOME_ADVANTAGE) / 400)
            )

            X = np.array([mod_row[c] for c in feature_cols]).reshape(1, -1)
            pred = prediction_agent.predict(X)
            conf = confidence_agent.score(pred)
            cal_prob = conf["calibrated_probability"]

            results.append({
                "home_win_prob": cal_prob,
                "confidence": conf["confidence_label"],
                "actual_home_win": int(row.get("HOME_WIN", -1)),
            })
        except Exception:
            continue
    return results


# ─────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────

def evaluate_model(name, results, min_edge=0.0):
    """Compute and print all metrics for one model."""
    valid = [r for r in results if r.get("actual_home_win", -1) in (0, 1)]
    if not valid:
        print(f"  {name}: No valid results.")
        return {}

    y_true = np.array([r["actual_home_win"] for r in valid])
    y_prob = np.array([r["home_win_prob"] for r in valid])
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    home_rate = y_true.mean()
    baseline = max(home_rate, 1 - home_rate)
    units, bets = calculate_units(valid, min_edge=min_edge)
    roi_pct = roi(units, bets)
    avg_edge = float(np.mean(np.abs(y_prob - 0.5)))

    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"  Games evaluated : {len(valid)}")
    print(f"  Accuracy        : {acc:.4f}  (baseline: {baseline:.4f})")
    print(f"  vs Baseline     : {acc - baseline:+.4f}")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Avg Edge        : {avg_edge:.4f}")
    print(f"  Bets placed     : {bets}  (min edge: {min_edge})")
    print(f"  Units +/-       : {units:+.2f}")
    print(f"  ROI             : {roi_pct:+.2f}%")

    # Accuracy by confidence tier
    print(f"\n  Accuracy by confidence tier:")
    for tier in ["High", "Medium", "Low"]:
        tier_res = [r for r in valid if r.get("confidence") == tier]
        if tier_res:
            t_true = np.array([r["actual_home_win"] for r in tier_res])
            t_prob = np.array([r["home_win_prob"] for r in tier_res])
            t_pred = (t_prob >= 0.5).astype(int)
            t_acc = accuracy_score(t_true, t_pred)
            t_units, t_bets = calculate_units(tier_res, min_edge=min_edge)
            t_roi = roi(t_units, t_bets)
            print(f"    {tier:<8}: acc={t_acc:.4f}  units={t_units:+.2f}  roi={t_roi:+.2f}%  n={len(tier_res)}")

    return {
        "name": name,
        "accuracy": acc,
        "brier": brier,
        "units": units,
        "bets": bets,
        "roi": roi_pct,
        "avg_edge": avg_edge,
    }


def print_comparison(results_dict):
    """Print side-by-side comparison of all models."""
    print(f"\n\n  {'='*65}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"  {'='*65}")
    print(f"  {'Model':<40} {'Acc':>6}  {'Units':>7}  {'ROI':>7}  {'Brier':>7}")
    print(f"  {'─'*65}")

    for name, m in results_dict.items():
        if m:
            print(
                f"  {name:<40} {m['accuracy']:.4f}  "
                f"{m['units']:>+7.2f}  {m['roi']:>+6.2f}%  {m['brier']:.4f}"
            )

    print(f"  {'='*65}\n")

    # Winner
    best = max(results_dict.items(), key=lambda x: x[1].get("roi", -999) if x[1] else -999)
    print(f"  Best ROI: {best[0]}")
    print(f"  {'='*65}\n")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest injury model versions")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to test")
    parser.add_argument("--model", type=str, choices=["A", "B", "C"], help="Run one model only")
    parser.add_argument("--min-edge", type=float, default=0.0, help="Min edge to place a bet")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  INJURY MODEL BACKTEST")
    print(f"  Season: {args.season}  |  Min edge: {args.min_edge}")
    print("=" * 65)

    test_df, feature_cols, strength_agent, prediction_agent = load_pipeline(args.season)
    if test_df is None:
        return

    confidence_agent = ConfidenceAgent()
    all_results = {}

    models_to_run = [args.model] if args.model else ["A", "B", "C"]

    if "A" in models_to_run:
        print("\n  Running Model A — No Injury Adjustment...")
        res_a = run_model_a(test_df, feature_cols, prediction_agent, confidence_agent)
        all_results["Model A — No Injury"] = evaluate_model(
            "Model A — No Injury Adjustment", res_a, args.min_edge
        )

    if "B" in models_to_run:
        print("\n  Running Model B — Fixed Penalty (flat -140)...")
        res_b = run_model_b(test_df, feature_cols, prediction_agent, confidence_agent, strength_agent)
        all_results["Model B — Fixed Penalty (-140)"] = evaluate_model(
            "Model B — Fixed Penalty (flat -140)", res_b, args.min_edge
        )

    if "C" in models_to_run:
        print("\n  Running Model C — Phase 3 Calibrated Tiers...")
        res_c = run_model_c(test_df, feature_cols, prediction_agent, confidence_agent)
        all_results["Model C — Calibrated Tiers"] = evaluate_model(
            "Model C — Phase 3 Calibrated Tiers", res_c, args.min_edge
        )

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()