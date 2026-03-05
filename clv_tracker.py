"""
clv_tracker.py — Closing Line Value tracker

Fetches opening and closing odds from The Odds API,
stores them alongside predictions, and measures whether
your model consistently beats the closing line.

Setup:
    1. Get a free API key at https://the-odds-api.com (500 requests/month free)
    2. Add your key to config.py:  ODDS_API_KEY = "your_key_here"
    OR set environment variable:   export ODDS_API_KEY=your_key_here

Usage:
    python clv_tracker.py --fetch          # fetch today's opening lines
    python clv_tracker.py --close          # fetch closing lines before tip-off
    python clv_tracker.py --summary        # show CLV analysis
    python clv_tracker.py --date 2026-03-03  # CLV for a specific date
"""

import json
import os
import argparse
import urllib.request
import urllib.parse
import time
from datetime import date, datetime, timezone
from pathlib import Path

LOG_FILE = Path("predictions_log.json")
ODDS_FILE = Path("cache/odds_log.json")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Get API key ──────────────────────────────────────────────────
def get_api_key():
    """Get Odds API key from config or environment."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    try:
        import config
        return getattr(config, "ODDS_API_KEY", None)
    except ImportError:
        return None


# ── Team name normalization ──────────────────────────────────────
# The Odds API uses full team names — map to abbreviations
ODDS_TEAM_MAP = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}


# ── Odds math ────────────────────────────────────────────────────

def american_to_prob(odds):
    """Convert American odds to implied probability."""
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def prob_to_american(prob):
    """Convert probability to American odds."""
    prob = float(prob)
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)


def remove_vig(home_prob, away_prob):
    """Remove bookmaker vig to get true implied probabilities."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


# ── API fetching ─────────────────────────────────────────────────

def fetch_odds(api_key, game_date=None):
    """
    Fetch NBA moneyline odds from The Odds API.

    Returns list of game odds dicts.
    Free tier: h2h (moneyline) markets only.
    """
    if game_date is None:
        game_date = date.today()

    date_str = str(game_date)

    # Build API URL
    # commence_time_from/to filters games on the target date
    params = {
        "apiKey":       api_key,
        "regions":      "us",
        "markets":      "h2h",
        "oddsFormat":   "american",
        "dateFormat":   "iso",
        "commenceTimeFrom": f"{date_str}T00:00:00Z",
        "commenceTimeTo":   f"{date_str}T23:59:59Z",
    }

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            remaining = resp.headers.get("x-requests-remaining", "?")
            print(f"  Odds API requests remaining: {remaining}")
            return data
    except Exception as e:
        print(f"  Error fetching odds: {e}")
        return []


def parse_game_odds(api_response):
    """
    Parse Odds API response into a clean dict per game.

    Returns: {
        "home_abbr@away_abbr": {
            "home": abbr,
            "away": abbr,
            "home_open": prob,
            "away_open": prob,
            "home_close": prob,  # filled in later
            "away_close": prob,
            "bookmakers": [...]
        }
    }
    """
    games = {}

    for game in api_response:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        home_abbr = ODDS_TEAM_MAP.get(home_team)
        away_abbr = ODDS_TEAM_MAP.get(away_team)

        if not home_abbr or not away_abbr:
            continue

        game_key = f"{home_abbr}@{away_abbr}"
        commence = game.get("commence_time", "")

        # Average odds across bookmakers for more accurate line
        home_probs = []
        away_probs = []

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    prob = american_to_prob(price)

                    if ODDS_TEAM_MAP.get(name) == home_abbr:
                        home_probs.append(prob)
                    elif ODDS_TEAM_MAP.get(name) == away_abbr:
                        away_probs.append(prob)

        if not home_probs or not away_probs:
            continue

        # Average across books
        raw_home = sum(home_probs) / len(home_probs)
        raw_away = sum(away_probs) / len(away_probs)

        # Remove vig
        home_true, away_true = remove_vig(raw_home, raw_away)

        games[game_key] = {
            "home":         home_abbr,
            "away":         away_abbr,
            "commence":     commence,
            "home_open":    round(home_true, 4),
            "away_open":    round(away_true, 4),
            "home_close":   None,
            "away_close":   None,
            "raw_home":     round(raw_home, 4),
            "raw_away":     round(raw_away, 4),
            "books_count":  len(home_probs),
        }

    return games


# ── Odds log management ──────────────────────────────────────────

def load_odds_log():
    if not ODDS_FILE.exists():
        return {}
    with open(ODDS_FILE) as f:
        return json.load(f)


def save_odds_log(data):
    with open(ODDS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def store_opening_lines(game_date, games_odds):
    """Store opening lines for a date."""
    log = load_odds_log()
    date_str = str(game_date)

    if date_str not in log:
        log[date_str] = {}

    for game_key, odds in games_odds.items():
        if game_key not in log[date_str]:
            log[date_str][game_key] = odds
            log[date_str][game_key]["fetched_at"] = datetime.now(timezone.utc).isoformat()
        else:
            # Preserve existing open, update close
            log[date_str][game_key]["home_close"] = odds["home_open"]
            log[date_str][game_key]["away_close"] = odds["away_open"]
            log[date_str][game_key]["closed_at"] = datetime.now(timezone.utc).isoformat()

    save_odds_log(log)
    return log[date_str]


# ── CLV calculation ──────────────────────────────────────────────

def calculate_clv(model_prob, closing_prob):
    """
    Calculate CLV as model probability minus closing line probability.
    Positive = model found value the market underpriced.
    Negative = market knew something the model missed.
    """
    return round(model_prob - closing_prob, 4)


def attach_clv_to_log(game_date=None):
    """
    Match odds log to predictions log and attach CLV values.
    Updates predictions_log.json with clv fields.
    """
    if game_date is None:
        game_date = date.today()

    date_str = str(game_date)
    odds_log = load_odds_log()
    date_odds = odds_log.get(date_str, {})

    if not date_odds:
        print(f"  No odds data found for {date_str}")
        return

    predictions = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    updated = 0

    for entry in predictions:
        if entry.get("date") != date_str:
            continue

        home = entry["home"]
        away = entry["away"]
        game_key = f"{home}@{away}"

        odds = date_odds.get(game_key)
        if not odds:
            continue

        # Use closing line if available, fall back to opening
        home_market_prob = odds.get("home_close") or odds.get("home_open")
        away_market_prob = odds.get("away_close") or odds.get("away_open")

        if not home_market_prob:
            continue

        model_home_prob = entry.get("home_win_prob", 0.5)
        line_type = "close" if odds.get("home_close") else "open"

        # CLV from home team perspective
        clv = calculate_clv(model_home_prob, home_market_prob)

        # CLV from predicted winner perspective
        if entry.get("predicted_winner") == home:
            winner_clv = calculate_clv(model_home_prob, home_market_prob)
        else:
            winner_clv = calculate_clv(1 - model_home_prob, away_market_prob)

        entry["market_home_prob"] = round(home_market_prob, 4)
        entry["market_away_prob"] = round(away_market_prob, 4)
        entry["home_clv"] = clv
        entry["winner_clv"] = winner_clv
        entry["line_type"] = line_type
        entry["market_line"] = prob_to_american(home_market_prob)
        updated += 1

    if updated:
        LOG_FILE.write_text(json.dumps(predictions, indent=2))
        print(f"  Attached CLV to {updated} predictions")


# ── CLV analysis ─────────────────────────────────────────────────

def show_clv_summary(game_date=None):
    """Show CLV analysis across all logged predictions."""
    predictions = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []

    if game_date:
        predictions = [p for p in predictions if p.get("date") == str(game_date)]

    # Only predictions with CLV data
    clv_preds = [p for p in predictions if p.get("winner_clv") is not None]

    if not clv_preds:
        print("  No CLV data yet. Run: python clv_tracker.py --fetch")
        return

    resolved = [p for p in clv_preds if p.get("actual_result") is not None]

    print(f"\n  CLV ANALYSIS")
    print(f"  {'─'*60}")
    print(f"  Predictions with CLV data : {len(clv_preds)}")
    print(f"  Resolved (result known)   : {len(resolved)}")

    if not resolved:
        print("  No resolved predictions with CLV data yet.")
        return

    # Overall CLV stats
    clvs = [p["winner_clv"] for p in resolved]
    avg_clv = sum(clvs) / len(clvs)
    positive_clv = sum(1 for c in clvs if c > 0)

    print(f"\n  Average CLV      : {avg_clv:+.3f}  ({avg_clv*100:+.1f}%)")
    print(f"  Positive CLV     : {positive_clv}/{len(clvs)} picks  ({100*positive_clv/len(clvs):.0f}%)")

    # CLV vs outcome correlation
    correct = [p for p in resolved if p.get("correct") == 1]
    wrong = [p for p in resolved if p.get("correct") == 0]

    if correct:
        avg_clv_correct = sum(p["winner_clv"] for p in correct) / len(correct)
        print(f"  Avg CLV (correct picks)  : {avg_clv_correct:+.3f}")
    if wrong:
        avg_clv_wrong = sum(p["winner_clv"] for p in wrong) / len(wrong)
        print(f"  Avg CLV (wrong picks)    : {avg_clv_wrong:+.3f}")

    # CLV buckets — where is the model finding edge?
    print(f"\n  CLV BREAKDOWN BY BUCKET:")
    print(f"  {'Range':<18} {'Picks':>6}  {'Win%':>6}  {'Avg Units':>10}  {'Interpretation'}")
    print(f"  {'─'*60}")

    buckets = [
        (0.05,  1.0,  "Strong edge"),
        (0.02,  0.05, "Mild edge"),
        (-0.02, 0.02, "No edge (noise)"),
        (-0.05, -0.02,"Market has edge"),
        (-1.0,  -0.05,"Market much better"),
    ]

    for lo, hi, label in buckets:
        bucket = [p for p in resolved if lo <= p["winner_clv"] < hi]
        if not bucket:
            continue
        wins = sum(1 for p in bucket if p.get("correct") == 1)
        units = sum(p.get("units", 0) for p in bucket)
        win_pct = wins / len(bucket) * 100
        avg_u = units / len(bucket)
        print(
            f"  [{lo:+.2f} to {hi:+.2f}]   "
            f"{len(bucket):>4}   {win_pct:>5.1f}%  {avg_u:>+9.3f}  {label}"
        )

    # Per-game CLV breakdown
    print(f"\n  RECENT GAMES — Model vs Market:")
    print(f"  {'Date':<12} {'Game':<14} {'Model':>7}  {'Market':>7}  {'CLV':>7}  {'Result'}")
    print(f"  {'─'*60}")

    for p in sorted(resolved, key=lambda x: x["date"])[-15:]:
        game = f"{p['away']}@{p['home']}"
        model_pct = f"{p['home_win_prob']*100:.1f}%"
        market_pct = f"{p.get('market_home_prob', 0)*100:.1f}%" if p.get('market_home_prob') else "N/A"
        clv_str = f"{p['winner_clv']:+.3f}" if p.get('winner_clv') is not None else "N/A"
        result = "WIN" if p.get("correct") == 1 else "LOSS"
        print(
            f"  {p['date']:<12} {game:<14} {model_pct:>7}  "
            f"{market_pct:>7}  {clv_str:>7}  {result}"
        )

    print(f"  {'─'*60}\n")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLV tracker for NBA predictions")
    parser.add_argument("--fetch", action="store_true", help="Fetch opening lines for today")
    parser.add_argument("--close", action="store_true", help="Fetch closing lines and attach CLV")
    parser.add_argument("--summary", action="store_true", help="Show CLV analysis")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()

    game_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()

    api_key = get_api_key()
    if not api_key and (args.fetch or args.close):
        print("\n  No Odds API key found.")
        print("  1. Get a free key at https://the-odds-api.com")
        print("  2. Add to config.py:  ODDS_API_KEY = 'your_key_here'")
        print("     or set env var:    export ODDS_API_KEY=your_key_here")
        return

    print(f"\n  CLV TRACKER — {game_date}")
    print(f"  {'─'*45}")

    if args.fetch:
        print(f"  Fetching opening lines for {game_date}...")
        raw = fetch_odds(api_key, game_date)
        if not raw:
            print("  No odds data returned.")
            return
        games = parse_game_odds(raw)
        stored = store_opening_lines(game_date, games)
        print(f"  Stored opening lines for {len(games)} games:")
        for key, odds in games.items():
            home_pct = f"{odds['home_open']*100:.1f}%"
            away_pct = f"{odds['away_open']*100:.1f}%"
            line = prob_to_american(odds['home_open'])
            print(f"    {key:<14}  Home: {home_pct}  Away: {away_pct}  Line: {line:+d}")

    elif args.close:
        print(f"  Fetching closing lines for {game_date}...")
        raw = fetch_odds(api_key, game_date)
        if not raw:
            print("  No odds data returned.")
            return
        games = parse_game_odds(raw)
        # Store as closing lines
        log = load_odds_log()
        date_str = str(game_date)
        if date_str not in log:
            log[date_str] = {}
        for game_key, odds in games.items():
            if game_key in log[date_str]:
                log[date_str][game_key]["home_close"] = odds["home_open"]
                log[date_str][game_key]["away_close"] = odds["away_open"]
                log[date_str][game_key]["closed_at"] = datetime.now(timezone.utc).isoformat()
            else:
                log[date_str][game_key] = odds
        save_odds_log(log)
        attach_clv_to_log(game_date)
        print(f"  Closing lines stored and CLV attached.")

    elif args.summary:
        show_clv_summary(
            game_date=game_date if args.date else None
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()