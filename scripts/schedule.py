import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import json
import requests as req
from datetime import date, datetime

from nba_api.stats.static import teams as nba_teams

from src.agents.orchestrator import Orchestrator, format_prediction
from src.agents.data_agent import DataAgent
from src.agents.team_strength_agent import TeamStrengthAgent
from src.agents.matchup_agent import MatchupAgent
from src.agents.prediction_agent import PredictionAgent

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

# ESPN uses slightly different abbreviations for some teams
ESPN_ABBR_MAP = {
    "GS":   "GSW",
    "SA":   "SAS",
    "NY":   "NYK",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
}

LOG_FILE = Path("predictions_log.json")
CACHE_DIR = Path("cache")


def _espn_to_nba_abbr(espn_abbr):
    """Convert ESPN team abbreviation to nba_api standard."""
    return ESPN_ABBR_MAP.get(espn_abbr, espn_abbr)


# ─────────────────────────────────────────────────────────────────
#  Pipeline loader
# ─────────────────────────────────────────────────────────────────

def load_pipeline():
    import pandas as pd
    import config

    cache_path = Path("cache/matchup_dataset.parquet")
    if not cache_path.exists():
        print("No cached data found. Run python run.py train first.")
        sys.exit(1)

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

    orch = Orchestrator(strength_agent=strength_agent)
    orch.prediction_agent = prediction_agent

    return orch, dataset, feature_cols, strength_agent


# ─────────────────────────────────────────────────────────────────
#  ESPN API — any date, no key needed, reliable
# ─────────────────────────────────────────────────────────────────

def fetch_espn_scoreboard(game_date):
    """Fetch raw ESPN scoreboard data for a given date."""
    date_str = game_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"

    try:
        resp = req.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception as e:
        print(f"  ESPN API failed: {e}")
        return []


def fetch_todays_games_espn(game_date=None):
    """Fetch schedule from ESPN API (any date)."""
    if game_date is None:
        game_date = date.today()

    print("  Using ESPN API...")
    events = fetch_espn_scoreboard(game_date)
    if not events:
        return []

    date_str = game_date.strftime("%Y-%m-%d")
    all_teams = nba_teams.get_teams()
    abbr_to_name = {t["abbreviation"]: t["full_name"] for t in all_teams}

    games = []
    for event in events:
        comp = event["competitions"][0]
        status_type = comp["status"]["type"]

        status_id = 1
        if status_type.get("completed", False):
            status_id = 3
        elif status_type.get("name") == "STATUS_IN_PROGRESS":
            status_id = 2

        home_team = away_team = None
        for team_entry in comp["competitors"]:
            abbr = _espn_to_nba_abbr(team_entry["team"]["abbreviation"])
            if team_entry["homeAway"] == "home":
                home_team = abbr
            else:
                away_team = abbr

        if not home_team or not away_team:
            continue

        status_text = status_type.get("shortDetail", "TBD")

        games.append({
            "game_id":   event["id"],
            "date":      date_str,
            "home":      home_team,
            "away":      away_team,
            "home_name": abbr_to_name.get(home_team, home_team),
            "away_name": abbr_to_name.get(away_team, away_team),
            "time":      status_text,
            "status_id": status_id,
        })

    return games


def fetch_final_scores_espn(game_date=None):
    """Fetch final scores from ESPN API (any date)."""
    if game_date is None:
        game_date = date.today()

    print("  Using ESPN API...")
    events = fetch_espn_scoreboard(game_date)
    if not events:
        return {}

    scores = {}
    for event in events:
        comp = event["competitions"][0]

        if not comp["status"]["type"].get("completed", False):
            continue

        home_abbr = away_abbr = None
        home_pts = away_pts = 0

        for team_entry in comp["competitors"]:
            abbr = _espn_to_nba_abbr(team_entry["team"]["abbreviation"])
            pts = int(team_entry.get("score", 0) or 0)
            if team_entry["homeAway"] == "home":
                home_abbr, home_pts = abbr, pts
            else:
                away_abbr, away_pts = abbr, pts

        if not home_abbr or not away_abbr:
            continue

        scores[event["id"]] = {
            "home":       home_abbr,
            "away":       away_abbr,
            "home_score": home_pts,
            "away_score": away_pts,
            "home_won":   1 if home_pts > away_pts else 0,
        }

    return scores


# ─────────────────────────────────────────────────────────────────
#  Live CDN — fastest, today only
# ─────────────────────────────────────────────────────────────────

def fetch_todays_games_live(game_date=None):
    """Primary for today: cdn.nba.com (instant, reliable)."""
    try:
        from nba_api.live.nba.endpoints import scoreboard

        print("  Using live CDN endpoint...")
        board = scoreboard.ScoreBoard()
        games = board.get_dict()["scoreboard"]["games"]
        all_teams = nba_teams.get_teams()
        abbr_to_name = {t["abbreviation"]: t["full_name"] for t in all_teams}

        date_str = (game_date or date.today()).strftime("%Y-%m-%d")

        result = []
        for g in games:
            home_abbr = g["homeTeam"]["teamTricode"]
            away_abbr = g["awayTeam"]["teamTricode"]

            result.append({
                "game_id":   g["gameId"],
                "date":      date_str,
                "home":      home_abbr,
                "away":      away_abbr,
                "home_name": abbr_to_name.get(home_abbr, home_abbr),
                "away_name": abbr_to_name.get(away_abbr, away_abbr),
                "time":      g.get("gameStatusText", "TBD").strip(),
                "status_id": g.get("gameStatus", 1),
            })
        return result
    except Exception as e:
        print(f"  Live CDN failed: {e}")
        return []


def fetch_final_scores_live():
    """Primary for today: final scores from cdn.nba.com."""
    try:
        from nba_api.live.nba.endpoints import scoreboard

        print("  Using live CDN endpoint...")
        board = scoreboard.ScoreBoard()
        games = board.get_dict()["scoreboard"]["games"]

        scores = {}
        for g in games:
            if g["gameStatus"] != 3:
                continue

            scores[g["gameId"]] = {
                "home":       g["homeTeam"]["teamTricode"],
                "away":       g["awayTeam"]["teamTricode"],
                "home_score": int(g["homeTeam"]["score"]),
                "away_score": int(g["awayTeam"]["score"]),
                "home_won":   1 if int(g["homeTeam"]["score"]) > int(g["awayTeam"]["score"]) else 0,
            }
        return scores
    except Exception as e:
        print(f"  Live CDN failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────
#  stats.nba.com — last resort
# ─────────────────────────────────────────────────────────────────

def fetch_todays_games_stats(game_date=None):
    """Last resort: stats.nba.com (slow, flaky)."""
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d") if hasattr(game_date, "strftime") else game_date
    print(f"  Falling back to stats.nba.com for {date_str}...")

    try:
        from nba_api.stats.endpoints import ScoreboardV3

        scoreboard = None
        for attempt in range(3):
            try:
                scoreboard = ScoreboardV3(
                    game_date=date_str,
                    league_id="00",
                    headers=HEADERS,
                    timeout=180,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Timeout, retrying ({attempt+1}/3)...")
                    time.sleep(15)
                else:
                    raise e
        time.sleep(0.6)

        dfs = scoreboard.get_data_frames()
        game_df = dfs[1]

        if game_df.empty:
            return []

        all_teams = nba_teams.get_teams()
        abbr_to_name = {t["abbreviation"]: t["full_name"] for t in all_teams}

        games = []
        for _, row in game_df.iterrows():
            code = str(row.get("gameCode", ""))
            if "/" not in code:
                continue
            teams_part = code.split("/")[1]
            if len(teams_part) != 6:
                continue
            away_abbr = teams_part[:3]
            home_abbr = teams_part[3:]

            games.append({
                "game_id":   str(row["gameId"]),
                "date":      date_str,
                "home":      home_abbr,
                "away":      away_abbr,
                "home_name": abbr_to_name.get(home_abbr, home_abbr),
                "away_name": abbr_to_name.get(away_abbr, away_abbr),
                "time":      str(row.get("gameStatusText", "TBD")).strip(),
                "status_id": int(row.get("gameStatus", 1)),
            })

        seen = set()
        unique = []
        for g in games:
            key = (g["home"], g["away"])
            if key not in seen:
                seen.add(key)
                unique.append(g)
        return unique

    except Exception as e:
        print(f"  stats.nba.com also failed: {e}")
        return []


def fetch_final_scores_stats(game_date=None):
    """Last resort: final scores from stats.nba.com."""
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d") if hasattr(game_date, "strftime") else game_date
    print(f"  Falling back to stats.nba.com for {date_str}...")

    try:
        from nba_api.stats.endpoints import ScoreboardV3

        scoreboard = None
        for attempt in range(3):
            try:
                scoreboard = ScoreboardV3(
                    game_date=date_str,
                    league_id="00",
                    headers=HEADERS,
                    timeout=180,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Timeout, retrying ({attempt+1}/3)...")
                    time.sleep(15)
                else:
                    raise e
        time.sleep(0.6)

        line_score = scoreboard.line_score.get_data_frame()
        game_header = scoreboard.game_header.get_data_frame()

        scores = {}
        for _, game_row in game_header.iterrows():
            game_id = str(game_row["gameId"])
            if int(game_row.get("gameStatus", 1)) != 3:
                continue

            code = str(game_row.get("gameCode", ""))
            if "/" not in code:
                continue
            teams_part = code.split("/")[1]
            if len(teams_part) != 6:
                continue
            away_abbr = teams_part[:3]
            home_abbr = teams_part[3:]

            game_lines = line_score[line_score["gameId"] == game_row["gameId"]]
            home_line = game_lines[game_lines["teamTricode"] == home_abbr]
            away_line = game_lines[game_lines["teamTricode"] == away_abbr]

            if home_line.empty or away_line.empty:
                continue

            scores[game_id] = {
                "home":       home_abbr,
                "away":       away_abbr,
                "home_score": int(home_line.iloc[0].get("score", 0) or 0),
                "away_score": int(away_line.iloc[0].get("score", 0) or 0),
                "home_won":   1 if int(home_line.iloc[0].get("score", 0) or 0) > int(away_line.iloc[0].get("score", 0) or 0) else 0,
            }
        return scores

    except Exception as e:
        print(f"  stats.nba.com also failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────
#  Unified fetchers — try sources in priority order
# ─────────────────────────────────────────────────────────────────

def fetch_todays_games(game_date=None):
    """1. Live CDN (today)  →  2. ESPN (any date)  →  3. stats.nba.com"""
    if game_date is None:
        game_date = date.today()

    if game_date == date.today():
        games = fetch_todays_games_live(game_date)
        if games:
            return games

    games = fetch_todays_games_espn(game_date)
    if games:
        return games

    return fetch_todays_games_stats(game_date)


def fetch_final_scores(game_date=None):
    """1. Live CDN (today)  →  2. ESPN (any date)  →  3. stats.nba.com"""
    if game_date is None:
        game_date = date.today()

    if game_date == date.today():
        scores = fetch_final_scores_live()
        if scores:
            return scores

    scores = fetch_final_scores_espn(game_date)
    if scores:
        return scores

    return fetch_final_scores_stats(game_date)


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
    log = load_log()
    date_str = str(game_date)
    return any(
        e["home"] == home and e["away"] == away and e["date"] == date_str
        for e in log
    )


# ─────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────

def print_schedule(games):
    print(f"\n  {'─'*58}")
    print(f"  {'Time':<10} {'Away':<22} {'Home':<22}")
    print(f"  {'─'*58}")
    for g in games:
        print(f"  {g['time']:<10} {g['away_name']:<22} {g['home_name']:<22}")
    print(f"  {'─'*58}")
    print(f"  {len(games)} game(s) scheduled\n")


def print_prediction_summary(game, pred):
    winner = pred["winner"]
    prob = round(pred["win_probability"] * 100, 1)
    conf = pred["confidence"]
    spread = pred["spread"]
    conf_score = pred["confidence_score"]

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
#  Auto-result filling
# ─────────────────────────────────────────────────────────────────

def auto_fill_results(game_date=None):
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
                print(f"  {result_str}  {away}@{home}  Score: {score_str}  ({entry['units']:+.3f}u)")
                updated += 1
                break

    if updated:
        save_log(log)
        print(f"\n  Updated {updated} result(s) in {LOG_FILE}")
    else:
        print("  No matching pending predictions found.")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-fetch NBA schedule and predict games")
    parser.add_argument("--date",    type=str,       help="Date (YYYY-MM-DD), default=today")
    parser.add_argument("--preview", action="store_true", help="Show schedule without predicting")
    parser.add_argument("--results", action="store_true", help="Auto-fill results from scoreboard")
    parser.add_argument("--summary", action="store_true", help="Show running prediction ROI")
    args = parser.parse_args()

    game_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()

    print(f"\n  {'='*58}")
    print(f"  NBA SCHEDULE — {game_date.strftime('%A, %B %d, %Y').replace(' 0', ' ')}")
    print(f"  {'='*58}")

    if args.results:
        auto_fill_results(game_date)
        return

    if args.summary:
        from scripts.prediction_tracker import show_summary
        show_summary()
        return

    games = fetch_todays_games(game_date)
    if not games:
        print("  No games found.")
        return

    print_schedule(games)
    if args.preview:
        return

    # Auto-update injuries
    print("  Updating injury report from ESPN...")
    try:
        from scripts.injury_updater import update as update_injuries
        update_injuries(verbose=False)
        print("  Injuries updated.")
    except Exception as e:
        print(f"  Could not auto-update injuries: {e}")

    # Predict all games
    print(f"  Running predictions...\n")
    print(f"  {'─'*58}")

    orch, dataset, feature_cols, strength = load_pipeline()
    predicted = 0
    skipped = 0

    for game in games:
        home = game["home"]
        away = game["away"]

        if already_logged(home, away, game_date):
            print(f"  {away}@{home}  -- already logged, skipping")
            skipped += 1
            continue

        if game["status_id"] > 1:
            print(f"  {away}@{home}  -- {game['time']}, skipping")
            skipped += 1
            continue

        try:
            pred = orch.predict_matchup(home, away, dataset, feature_cols)
            if "error" in pred:
                print(f"  {away}@{home}  -- error: {pred['error']}")
                continue

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
    print(f"\n  Fill results tonight with:")
    print(f"  python run.py results")
    print(f"  {'='*58}\n")


if __name__ == "__main__":
    main()