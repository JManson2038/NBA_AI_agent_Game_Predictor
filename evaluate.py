"""
evaluate.py — Backtesting and evaluation metrics
Tracks accuracy, Brier score, calibration, and model agreement analysis.
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


def evaluate_predictions(results):
    """
    Evaluate a list of prediction results from the Orchestrator.
    """
    valid = [r for r in results if r.get("actual_home_win", -1) in (0, 1)]

    if not valid:
        print("No valid results to evaluate.")
        return

    y_true = np.array([r["actual_home_win"] for r in valid])
    y_prob = np.array([r["home_win_prob"] for r in valid])
    y_pred = (y_prob >= 0.5).astype(int)

    n = len(valid)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    home_win_rate = y_true.mean()
    baseline_acc = max(home_win_rate, 1 - home_win_rate)

    print("")
    print("=" * 55)
    print("  EVALUATION RESULTS (" + str(n) + " games)")
    print("=" * 55)
    print("  Accuracy:       " + str(round(acc, 4)) + "  (baseline: " + str(round(baseline_acc, 4)) + ")")
    print("  Log Loss:       " + str(round(ll, 4)))
    print("  Brier Score:    " + str(round(brier, 4)))
    print("  Home Win Rate:  " + str(round(home_win_rate, 4)))
    print("  vs Baseline:    " + str(round(acc - baseline_acc, 4)))

    # ── Accuracy by Confidence Tier ──
    print("")
    print("  ACCURACY BY CONFIDENCE TIER:")
    for tier in ["High", "Medium", "Low"]:
        tier_res = [r for r in valid if r.get("confidence") == tier]
        if tier_res:
            t_true = np.array([r["actual_home_win"] for r in tier_res])
            t_prob = np.array([r["home_win_prob"] for r in tier_res])
            t_pred = (t_prob >= 0.5).astype(int)
            t_acc = accuracy_score(t_true, t_pred)
            t_brier = brier_score_loss(t_true, t_prob)
            print("    " + tier.ljust(8) + ": acc=" + str(round(t_acc, 4))
                  + "  brier=" + str(round(t_brier, 4))
                  + "  n=" + str(len(tier_res)))
        else:
            print("    " + tier.ljust(8) + ": N/A")

    # ── Calibration (expected vs actual) ──
    print("")
    print("  CALIBRATION (predicted prob vs actual win rate):")
    buckets = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    for lo, hi in buckets:
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            actual = y_true[mask].mean()
            predicted = y_prob[mask].mean()
            diff = abs(actual - predicted)
            marker = " <-- well calibrated" if diff < 0.05 else ""
            print("    [" + str(lo) + "-" + str(hi) + "): "
                  + "pred=" + str(round(predicted, 3))
                  + "  actual=" + str(round(actual, 3))
                  + "  n=" + str(int(mask.sum())) + marker)

    # ── Model Conflict Analysis ──
    conflict_games = [r for r in valid
                      if r.get("confidence_breakdown", {}).get("models_conflict", False)]
    agree_games = [r for r in valid
                   if not r.get("confidence_breakdown", {}).get("models_conflict", False)]

    if conflict_games:
        print("")
        print("  MODEL CONFLICT ANALYSIS:")
        c_true = np.array([r["actual_home_win"] for r in conflict_games])
        c_pred = np.array([(r["home_win_prob"] >= 0.5) for r in conflict_games]).astype(int)
        c_acc = accuracy_score(c_true, c_pred)
        print("    Conflict:  acc=" + str(round(c_acc, 4)) + "  n=" + str(len(conflict_games)))
        if agree_games:
            a_true = np.array([r["actual_home_win"] for r in agree_games])
            a_pred = np.array([(r["home_win_prob"] >= 0.5) for r in agree_games]).astype(int)
            a_acc = accuracy_score(a_true, a_pred)
            print("    Agreement: acc=" + str(round(a_acc, 4)) + "  n=" + str(len(agree_games)))

    print("=" * 55)
    print("")

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier": brier,
    }