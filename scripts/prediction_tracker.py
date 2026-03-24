
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import date
from pathlib import Path
from src.agents.orchestrator import Orchestrator, format_prediction


LOG_FILE = Path("predictions_log.json")


def load_log():
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        return json.load(f)


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def log_prediction(home, away):
    """Run a prediction and save it with full context."""
    from main import load_pipeline
    from src.agents.orchestrator import format_prediction

    orch, dataset, feature_cols, strength = load_pipeline()
    pred = orch.predict_matchup(home, away, dataset, feature_cols)

    if "error" in pred:
        print(f"Error: {pred['error']}")
        return

    # Build log entry
    entry = {
        "date":            str(date.today()),
        "home":            home,
        "away":            away,
        "predicted_winner": pred["winner"],
        "home_win_prob":   pred["home_win_prob"],
        "win_probability": pred["win_probability"],
        "confidence":      pred["confidence"],
        "confidence_score": pred["confidence_score"],
        "spread":          pred["spread"],
        "key_factors":     pred["key_factors"],
        "injury_report":   pred.get("injury_report", {}),
        "model_detail":    pred["model_detail"],
        "actual_result":   None,   # filled in later
        "correct":         None,
        "units":           None,
    }

    log = load_log()
    log.append(entry)
    save_log(log)

    print(format_prediction(pred))
    print(f"\n  Saved to {LOG_FILE}")
    print(f"  Record result with: python prediction_tracker.py --result {home} {away} <1 or 0>")


def record_result(home, away, home_won):
    """Record the actual result of a previously logged game."""
    log = load_log()
    matched = False

    for entry in reversed(log):
        if entry["home"] == home and entry["away"] == away and entry["actual_result"] is None:
            entry["actual_result"] = home_won
            pred_home_win = entry["home_win_prob"] >= 0.5
            entry["correct"] = int(pred_home_win == bool(home_won))

            # Units at -110
            payout = 100 / 110
            pred_winner_won = (
                (entry["predicted_winner"] == home and home_won == 1) or
                (entry["predicted_winner"] == away and home_won == 0)
            )
            entry["units"] = round(payout if pred_winner_won else -1.0, 4)

            matched = True
            break

    if not matched:
        print(f"No pending prediction found for {away} @ {home}")
        return

    save_log(log)
    result_str = f"{home} wins" if home_won else f"{away} wins"
    print(f"  Result recorded: {result_str}")
    print(f"  Prediction was: {'CORRECT' if entry['correct'] else 'WRONG'}")
    print(f"  Units: {entry['units']:+.3f}")


def show_summary(min_confidence=None):
    """Show ROI, accuracy, and breakdown by confidence tier."""
    log = load_log()
    resolved = [e for e in log if e["actual_result"] is not None]

    if not resolved:
        print("No resolved predictions yet.")
        return

    if min_confidence:
        resolved = [e for e in resolved if e["confidence"] == min_confidence]

    total = len(resolved)
    correct = sum(e["correct"] for e in resolved)
    units = sum(e["units"] for e in resolved)
    roi = (units / total) * 100 if total else 0

    print(f"\n  PREDICTION TRACKER SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total predictions : {total}")
    print(f"  Correct           : {correct} ({100*correct/total:.1f}%)")
    print(f"  Units +/-         : {units:+.2f}")
    print(f"  ROI               : {roi:+.2f}%")

    # By confidence
    print(f"\n  By confidence tier:")
    for tier in ["High", "Medium", "Low"]:
        tier_games = [e for e in resolved if e["confidence"] == tier]
        if tier_games:
            t_correct = sum(e["correct"] for e in tier_games)
            t_units = sum(e["units"] for e in tier_games)
            t_roi = (t_units / len(tier_games)) * 100
            print(
                f"    {tier:<8}: {t_correct}/{len(tier_games)} correct  "
                f"units={t_units:+.2f}  roi={t_roi:+.2f}%"
            )

    # By date
    print(f"\n  Recent results (last 10):")
    print(f"  {'Date':<12} {'Matchup':<18} {'Pred':<6} {'Prob':>6}  {'Conf':<8} {'Result'}")
    print(f"  {'─'*65}")
    for e in reversed(resolved[-10:]):
        matchup = f"{e['away']}@{e['home']}"
        result = "WIN" if e["correct"] else "LOSS"
        print(
            f"  {e['date']:<12} {matchup:<18} {e['predicted_winner']:<6} "
            f"{e['win_probability']:>5.1%}  {e['confidence']:<8} {result} ({e['units']:+.02f}u)"
        )


def show_pending():
    """Show predictions waiting for results."""
    log = load_log()
    pending = [e for e in log if e["actual_result"] is None]

    if not pending:
        print("No pending predictions.")
        return

    print(f"\n  PENDING PREDICTIONS ({len(pending)})")
    print(f"  {'─'*60}")
    print(f"  {'Date':<12} {'Matchup':<18} {'Pick':<6} {'Prob':>6}  {'Conf'}")
    print(f"  {'─'*60}")
    for e in pending:
        matchup = f"{e['away']}@{e['home']}"
        print(
            f"  {e['date']:<12} {matchup:<18} {e['predicted_winner']:<6} "
            f"{e['win_probability']:>5.1%}  {e['confidence']}"
        )
    print(f"\n  Record with: python prediction_tracker.py --result HOME AWAY <1=home wins, 0=away wins>")


def main():
    parser = argparse.ArgumentParser(description="Track live NBA predictions")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="Log a new prediction")
    parser.add_argument("--result", nargs=3, metavar=("HOME", "AWAY", "RESULT"), help="Record result (1=home win, 0=away win)")
    parser.add_argument("--summary", action="store_true", help="Show ROI summary")
    parser.add_argument("--pending", action="store_true", help="Show pending predictions")
    parser.add_argument("--confidence", type=str, help="Filter summary by confidence tier")
    args = parser.parse_args()

    if args.predict:
        log_prediction(args.predict[0].upper(), args.predict[1].upper())
    elif args.result:
        home, away, result = args.result
        log_prediction_result = int(result)
        record_result(home.upper(), away.upper(), log_prediction_result)
    elif args.summary:
        show_summary(min_confidence=args.confidence)
    elif args.pending:
        show_pending()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()