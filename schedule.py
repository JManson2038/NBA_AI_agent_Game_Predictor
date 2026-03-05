"""
schedule.py — Auto-fetch today's NBA schedule and log predictions

Pulls today's games from nba_api, runs predictions for each,
and saves them to predictions_log.json ready for result entry.

Usage:
    python schedule.py                  # predict all of today's games
    python schedule.py --date 2026-03-04  # predict a specific date
    python schedule.py --preview        # show today's games without predicting
    python schedule.py --results        # prompt to enter results for pending games
"""

import argparse
import time
import json
from datetime import date, datetime
from pathlib import Path

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams

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

LOG_FILE = Path("predictions_log.json")
CACHE_DIR = Path("cache")


# ─────────────────────────────────────────────────────────────────
#  Schedule fetching
# ─────────────────────────────────────────────────────────────────

def fetch_todays_games(game_date=None):
    """
    Fetch NBA games for a given date using ScoreboardV2.
    Returns list of dicts with home/away team abbreviations,
    game time, and current status.
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d") if hasattr(game_date, "strftime") else game_date

    print(f"  Fetching schedule for {date_str}...")

    try:
        scoreboard = None
        for attempt in range(3):
            try:
                scoreboard = ScoreboardV2(
                    game_date=date_str,
                    league_id="00",
                    day_offset=0,
                    headers=HEADERS,
                    timeout=60,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Timeout, retrying ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    raise e
        time.sleep(0.6)

        # Build team ID -> abbreviation map
        all_teams = nba_teams.get_teams()
        id_to_abbr = {t["id"]: t["abbreviation"] for t in all_teams}
        id_to_name = {t["id"]: t["full_name"] for t in all_teams}

        # GameHeader has one row per game
        game_header = scoreboard.game_header.get_data_frame()

        if game_header.empty:
            print(f"  No games scheduled for {date_str}.")
            return []

        games = []
        for _, row in game_header.iterrows():
            home_id = int(row["HOME_TEAM_ID"])
            away_id = int(row["VISITOR_TEAM_ID"])
            game_id = str(row["GAME_ID"])
            game_time = str(row.get("GAME_STATUS_TEXT", "TBD")).strip()
            status_id = int(row.get("GAME_STATUS_ID", 1))

            home_abbr = id_to_abbr.get(home_id, str(home_id))
            away_abbr = id_to_abbr.get(away_id, str(away_id))
            home_name = id_to_name.get(home_id, home_abbr)
            away_name = id_to_name.get(away_id, away_abbr)

            games.append({
                "game_id":   game_id,
                "date":      date_str,
                "home":      home_abbr,
                "away":      away_abbr,
                "home_name": home_name,
                "away_name": away_name,
                "time":      game_time,
                "status_id": status_id,
                # status_id: 1=scheduled, 2=in progress, 3=final
            })

        # Deduplicate — same home/away pair can appear twice in API response
        seen = set()
        unique_games = []
        for g in games:
            key = (g["home"], g["away"])
            if key not in seen:
                seen.add(key)
                unique_games.append(g)

        return unique_games

    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return []


def fetch_final_scores(game_date=None):
    """
    Fetch final scores for completed games on a given date.
    Returns dict of {game_id: {home_score, away_score, home_won}}
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d") if hasattr(game_date, "strftime") else game_date

    try:
        scoreboard = None
        for attempt in range(3):
            try:
                scoreboard = ScoreboardV2(
                    game_date=date_str,
                    league_id="00",
                    day_offset=0,
                    headers=HEADERS,
                    timeout=60,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Timeout, retrying ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    raise e
        time.sleep(0.6)

        line_score = scoreboard.line_score.get_data_frame()
        game_header = scoreboard.game_header.get_data_frame()

        all_teams = nba_teams.get_teams()
        id_to_abbr = {t["id"]: t["abbreviation"] for t in all_teams}

        scores = {}
        for _, game_row in game_header.iterrows():
            game_id = str(game_row["GAME_ID"])
            status_id = int(game_row.get("GAME_STATUS_ID", 1))

            if status_id != 3:
                continue  # Not final yet

            home_id = int(game_row["HOME_TEAM_ID"])
            away_id = int(game_row["VISITOR_TEAM_ID"])

            # Find scores in line_score
            home_line = line_score[line_score["TEAM_ID"] == home_id]
            away_line = line_score[line_score["TEAM_ID"] == away_id]

            if home_line.empty or away_line.empty:
                continue

            home_pts = int(home_line.iloc[0].get("PTS", 0) or 0)
            away_pts = int(away_line.iloc[0].get("PTS", 0) or 0)

            scores[game_id] = {
                "home":      id_to_abbr.get(home_id, str(home_id)),
                "away":      id_to_abbr.get(away_id, str(away_id)),
                "home_score": home_pts,
                "away_score": away_pts,
                "home_won":  1 if home_pts > away_pts else 0,
            }

        return scores

    except Exception as e:
        print(f"  Error fetching scores: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────
#  Prediction logging
# ─────────────────────────────────────────────────────────────────

def load_log():
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        return json.load(f)


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def already_logged(home, away, game_date):
    """Check if this game is already in the prediction log."""
    log = load_log()
    date_str = str(game_date)
    return any(
        e["home"] == home and e["away"] == away and e["date"] == date_str
        for e in log
    )


def log_prediction(home, away, game_date, game_id=None):
    """Run prediction and save to log."""
    from main import load_pipeline
    from orchestrator import format_prediction

    orch, dataset, feature_cols, strength = load_pipeline()
    pred = orch.predict_matchup(home, away, dataset, feature_cols)

    if "error" in pred:
        print(f"  Error predicting {away}@{home}: {pred['error']}")
        return None

    entry = {
        "game_id":          game_id,
        "date":             str(game_date),
        "home":             home,
        "away":             away,
        "predicted_winner": pred["winner"],
        "home_win_prob":    pred["home_win_prob"],
        "win_probability":  pred["win_probability"],
        "confidence":       pred["confidence"],
        "confidence_score": pred["confidence_score"],
        "spread":           pred["spread"],
        "key_factors":      pred["key_factors"],
        "injury_report":    pred.get("injury_report", {}),
        "model_detail":     pred["model_detail"],
        "actual_result":    None,
        "correct":          None,
        "units":            None,
    }

    log = load_log()
    log.append(entry)
    save_log(log)

    return pred


def record_result(home, away, home_won, game_date=None):
    """Record the actual result for a logged game."""
    log = load_log()
    date_str = str(game_date) if game_date else None
    matched = False

    for entry in reversed(log):
        date_match = (date_str is None) or (entry["date"] == date_str)
        if entry["home"] == home and entry["away"] == away and \
           entry["actual_result"] is None and date_match:

            entry["actual_result"] = home_won
            pred_home_win = entry["home_win_prob"] >= 0.5
            entry["correct"] = int(pred_home_win == bool(home_won))

            payout = 100 / 110
            pred_winner_won = (
                (entry["predicted_winner"] == home and home_won == 1) or
                (entry["predicted_winner"] == away and home_won == 0)
            )
            entry["units"] = round(payout if pred_winner_won else -1.0, 4)
            matched = True
            break

    if matched:
        save_log(log)
    return matched


# ─────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────

def print_schedule(games):
    """Print today's schedule in a clean table."""
    print(f"\n  {'─'*58}")
    print(f"  {'Time':<10} {'Away':<22} {'Home':<22}")
    print(f"  {'─'*58}")
    for g in games:
        status = g["time"]
        print(f"  {status:<10} {g['away_name']:<22} {g['home_name']:<22}")
    print(f"  {'─'*58}")
    print(f"  {len(games)} game(s) scheduled\n")


def print_prediction_summary(game, pred):
    """Print a compact one-line prediction summary."""
    winner = pred["winner"]
    prob = round(pred["win_probability"] * 100, 1)
    conf = pred["confidence"]
    spread = pred["spread"]
    conf_score = pred["confidence_score"]

    # Injury flags
    inj = pred.get("injury_report", {})
    home_pen = inj.get("home", {}).get("impact", 0)
    away_pen = inj.get("away", {}).get("impact", 0)
    inj_flag = ""
    if home_pen > 30 or away_pen > 30:
        inj_flag = f"  [INJ: -{round(home_pen)}H/-{round(away_pen)}A]"

    print(
        f"  {game['away']:<4} @ {game['home']:<4}  "
        f"Pick: {winner:<4}  {prob:.1f}%  "
        f"Spread: {spread:+.1f}  "
        f"Conf: {conf} ({conf_score}){inj_flag}"
    )


# ─────────────────────────────────────────────────────────────────
#  Auto-result entry from scoreboard
# ─────────────────────────────────────────────────────────────────

def auto_fill_results(game_date=None):
    """
    Fetch final scores for a date and auto-fill results
    into the prediction log without manual entry.
    """
    if game_date is None:
        game_date = date.today()

    print(f"\n  Checking final scores for {game_date}...")
    scores = fetch_final_scores(game_date)

    if not scores:
        print("  No final scores found yet.")
        return

    log = load_log()
    updated = 0

    for game_id, result in scores.items():
        home = result["home"]
        away = result["away"]
        home_won = result["home_won"]

        for entry in reversed(log):
            if (entry["home"] == home and entry["away"] == away and
                    entry["actual_result"] is None and
                    entry["date"] == str(game_date)):

                entry["actual_result"] = home_won
                pred_home_win = entry["home_win_prob"] >= 0.5
                entry["correct"] = int(pred_home_win == bool(home_won))

                payout = 100 / 110
                pred_winner_won = (
                    (entry["predicted_winner"] == home and home_won == 1) or
                    (entry["predicted_winner"] == away and home_won == 0)
                )
                entry["units"] = round(payout if pred_winner_won else -1.0, 4)

                result_str = "WIN " if entry["correct"] else "LOSS"
                score_str = f"{result['away_score']}-{result['home_score']}"
                print(
                    f"  {result_str}  {away}@{home}  "
                    f"Score: {score_str}  "
                    f"({entry['units']:+.3f}u)"
                )
                updated += 1
                break

    if updated:
        save_log(log)
        print(f"\n  Updated {updated} result(s) in {LOG_FILE}")

        # ── Auto-attach CLV closing lines ──
        try:
            from clv_tracker import fetch_odds, parse_game_odds, store_opening_lines, attach_clv_to_log, get_api_key
            api_key = get_api_key()
            if api_key:
                print(f"  Fetching closing lines for CLV...")
                raw = fetch_odds(api_key, game_date)
                if raw:
                    games_odds = parse_game_odds(raw)
                    store_opening_lines(game_date, games_odds)
                    attach_clv_to_log(game_date)
        except Exception as e:
            print(f"  Could not attach CLV: {e}")
    else:
        print("  No matching pending predictions found.")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-fetch NBA schedule and predict games")
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD), default=today")
    parser.add_argument("--preview", action="store_true", help="Show schedule without predicting")
    parser.add_argument("--results", action="store_true", help="Auto-fill results from scoreboard")
    parser.add_argument("--summary", action="store_true", help="Show running prediction ROI")
    args = parser.parse_args()

    # Parse date
    if args.date:
        game_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        game_date = date.today()

    print(f"\n  {'='*58}")
    print(f"  NBA SCHEDULE — {game_date.strftime('%A, %B %-d, %Y')}")
    print(f"  {'='*58}")

    # ── Auto-fill results mode ──
    if args.results:
        auto_fill_results(game_date)
        return

    # ── Summary mode ──
    if args.summary:
        from prediction_tracker import show_summary
        show_summary()
        return

    # ── Fetch today's games ──
    games = fetch_todays_games(game_date)

    if not games:
        print("  No games found.")
        return

    print_schedule(games)

    if args.preview:
        return

    # ── Auto-update injuries before predicting ──
    print("  Updating injury report from ESPN...")
    try:
        from injury_updater import update as update_injuries
        update_injuries(verbose=False)
        print("  Injuries updated.")
    except Exception as e:
        print(f"  Could not auto-update injuries: {e}")

    # ── Predict all games ──
    print(f"  Running predictions...\n")
    print(f"  {'─'*58}")

    from main import load_pipeline
    orch, dataset, feature_cols, strength = load_pipeline()

    predicted = 0
    skipped = 0

    for game in games:
        home = game["home"]
        away = game["away"]

        # Skip games already logged today
        if already_logged(home, away, game_date):
            print(f"  {away}@{home}  -- already logged, skipping")
            skipped += 1
            continue

        # Skip games already in progress or finished
        if game["status_id"] > 1:
            print(f"  {away}@{home}  -- {game['time']}, skipping")
            skipped += 1
            continue

        try:
            from orchestrator import format_prediction
            pred = orch.predict_matchup(home, away, dataset, feature_cols)

            if "error" in pred:
                print(f"  {away}@{home}  -- error: {pred['error']}")
                continue

            # Save to log
            entry = {
                "game_id":          game["game_id"],
                "date":             str(game_date),
                "home":             home,
                "away":             away,
                "predicted_winner": pred["winner"],
                "home_win_prob":    pred["home_win_prob"],
                "win_probability":  pred["win_probability"],
                "confidence":       pred["confidence"],
                "confidence_score": pred["confidence_score"],
                "spread":           pred["spread"],
                "key_factors":      pred["key_factors"],
                "injury_report":    pred.get("injury_report", {}),
                "model_detail":     pred["model_detail"],
                "actual_result":    None,
                "correct":          None,
                "units":            None,
            }
            log = load_log()
            log.append(entry)
            save_log(log)

            print_prediction_summary(game, pred)
            predicted += 1

        except Exception as e:
            print(f"  {away}@{home}  -- failed: {e}")

    print(f"  {'─'*58}")
    print(f"  Predicted: {predicted}  Skipped: {skipped}")
    print(f"  Saved to: {LOG_FILE}")

    # ── Auto-fetch opening lines ──
    try:
        from clv_tracker import fetch_odds, parse_game_odds, store_opening_lines, get_api_key
        api_key = get_api_key()
        if api_key:
            print(f"\n  Fetching opening lines...")
            raw = fetch_odds(api_key, game_date)
            if raw:
                games_odds = parse_game_odds(raw)
                store_opening_lines(game_date, games_odds)
                print(f"  Opening lines stored for {len(games_odds)} games.")
        else:
            print(f"\n  No Odds API key — skipping opening lines.")
            print(f"  Add ODDS_API_KEY to config.py to enable CLV tracking.")
    except Exception as e:
        print(f"\n  Could not fetch opening lines: {e}")
    print(f"\n  Fill results tonight with:")
    print(f"  python schedule.py --results")
    print(f"  {'='*58}\n")


if __name__ == "__main__":
    main()