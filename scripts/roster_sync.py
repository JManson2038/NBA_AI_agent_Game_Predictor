"""
roster_sync.py — Sync all 30 NBA rosters and detect trades, signings, and releases.

Roster sources are tried in order, same pattern as schedule.py:
    1. ESPN site API   fast, public, no headers needed
    2. stats.nba.com   via nba_api CommonTeamRoster (slow, frequently times out)

Usage:
    python run.py roster                 # sync all 30 teams
    python run.py roster --team LAL      # sync one team
    python run.py roster --check         # check cache staleness only
    python run.py roster --trades        # show detected trades only
    python run.py roster --source nba    # force a single source (espn | nba)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
import unicodedata
import urllib.request
from datetime import datetime, timezone

try:
    import config
    SEASON = config.CURRENT_SEASON
except (ImportError, AttributeError):
    SEASON = "2026-27"

CACHE_DIR = Path("cache")
INJURY_FILE = Path("injuries.json")
ROSTER_CACHE = CACHE_DIR / "rosters.json"
SYNC_LOG = CACHE_DIR / "roster_sync.log"
STALE_DAYS = 3

ALL_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA", "WAS",
]

ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"

# ESPN uses slightly different abbreviations for some teams
ESPN_ABBR_MAP = {
    "GS":   "GSW",
    "SA":   "SAS",
    "NY":   "NYK",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
}

NBA_STATS_HEADERS = {
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


# ─────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────

def _normalize_name(name):
    """'Luka Dončić' -> 'luka doncic'. Used to match players across sources."""
    nfkd = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.replace(".", "").lower().split())


def _http_get_json(url, timeout=15, retries=3, delay=2):
    """GET a JSON URL with a browser User-Agent and simple retry."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_err


def _player_record(player_id, name, num, position, source):
    return {
        "id":       int(player_id) if str(player_id).isdigit() else str(player_id),
        "name":     str(name).strip(),
        "num":      str(num or "").strip(),
        "position": str(position or "").strip(),
        "source":   source,
    }


# ─────────────────────────────────────────────────────────────────
#  Source 1: ESPN
# ─────────────────────────────────────────────────────────────────

_espn_team_ids = None


def fetch_espn_team_ids():
    """Map NBA abbreviation -> ESPN numeric team id. Fetched once per run."""
    global _espn_team_ids
    if _espn_team_ids is not None:
        return _espn_team_ids

    data = _http_get_json(ESPN_TEAMS_URL)
    teams = data["sports"][0]["leagues"][0]["teams"]
    _espn_team_ids = {}
    for entry in teams:
        team = entry["team"]
        abbr = ESPN_ABBR_MAP.get(team["abbreviation"], team["abbreviation"])
        _espn_team_ids[abbr] = team["id"]
    return _espn_team_ids


def fetch_roster_espn(team_abbr):
    """Fetch one team's roster from the ESPN site API."""
    team_id = fetch_espn_team_ids().get(team_abbr)
    if team_id is None:
        raise ValueError(f"ESPN has no team id for {team_abbr}")

    data = _http_get_json(ESPN_ROSTER_URL.format(team_id=team_id))
    players = []
    for athlete in data.get("athletes", []):
        players.append(_player_record(
            player_id=athlete.get("id"),
            name=athlete.get("fullName") or athlete.get("displayName"),
            num=athlete.get("jersey"),
            position=(athlete.get("position") or {}).get("abbreviation"),
            source="espn",
        ))
    return players


# ─────────────────────────────────────────────────────────────────
#  Source 2: stats.nba.com (nba_api)
# ─────────────────────────────────────────────────────────────────

def fetch_roster_nba(team_abbr, retries=2, timeout=20):
    """Fetch one team's roster from stats.nba.com. Slow and flaky; used as fallback."""
    from nba_api.stats.static import teams as nba_teams
    from nba_api.stats.endpoints import CommonTeamRoster

    abbr_to_id = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
    team_id = abbr_to_id.get(team_abbr)
    if team_id is None:
        raise ValueError(f"nba_api has no team id for {team_abbr}")

    last_err = None
    for attempt in range(retries):
        try:
            roster = CommonTeamRoster(
                team_id=team_id,
                season=SEASON,
                headers=NBA_STATS_HEADERS,
                timeout=timeout,
            )
            df = roster.get_data_frames()[0]
            time.sleep(0.6)
            return [
                _player_record(
                    player_id=row["PLAYER_ID"],
                    name=row["PLAYER"],
                    num=row.get("NUM", ""),
                    position=row.get("POSITION", ""),
                    source="nba",
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(3)
    raise last_err


# ─────────────────────────────────────────────────────────────────
#  Source chain
# ─────────────────────────────────────────────────────────────────

SOURCES = {
    "espn": fetch_roster_espn,
    "nba":  fetch_roster_nba,
}
SOURCE_ORDER = ["espn", "nba"]


def fetch_team_roster(team_abbr, source="auto"):
    """
    Try each source in order until one returns players.
    Returns (players, source_name) or (None, last_error_message).
    """
    order = SOURCE_ORDER if source == "auto" else [source]
    errors = []
    for name in order:
        try:
            players = SOURCES[name](team_abbr)
            if players:
                return players, name
            errors.append(f"{name}: empty roster")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return None, " | ".join(errors)


def fetch_all_rosters(teams_to_sync, source="auto"):
    """Fetch rosters for a list of teams. Returns (rosters, failed_teams)."""
    rosters = {}
    failed = {}
    total = len(teams_to_sync)
    for i, abbr in enumerate(teams_to_sync, 1):
        print(f"  [{i:>2}/{total}] {abbr}...", end=" ", flush=True)
        players, info = fetch_team_roster(abbr, source)
        if players is None:
            failed[abbr] = info
            print(f"FAILED  ({info})")
        else:
            rosters[abbr] = players
            print(f"{len(players):>2} players  [{info}]")
    return rosters, failed


# ─────────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────────

def load_roster_cache():
    if ROSTER_CACHE.exists():
        with open(ROSTER_CACHE) as f:
            return json.load(f)
    return {}


def save_roster_cache(rosters):
    CACHE_DIR.mkdir(exist_ok=True)
    data = {
        "_last_updated": datetime.now(timezone.utc).isoformat(),
        "_season": SEASON,
    }
    data.update(rosters)
    with open(ROSTER_CACHE, "w") as f:
        json.dump(data, f, indent=2)


def check_stale(warn_only=False):
    """Return True if the roster cache is missing or older than STALE_DAYS."""
    if not ROSTER_CACHE.exists():
        print("  No roster cache found. Run: python run.py roster")
        return True

    last_updated = load_roster_cache().get("_last_updated")
    if not last_updated:
        print("  Roster cache has no timestamp. Run: python run.py roster")
        return True

    last_dt = datetime.fromisoformat(last_updated)
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - last_dt
    age_days = age.days
    age_hours = int(age.total_seconds() / 3600)

    if age_days >= STALE_DAYS:
        print(f"  WARNING: Roster cache is {age_days} days old (last updated: {last_updated[:10]})")
        print("  Run: python run.py roster  to get current rosters")
        return True

    if not warn_only:
        print(f"  Roster cache is fresh ({age_hours}h old, updated {last_updated[:10]})")
    return False


# ─────────────────────────────────────────────────────────────────
#  Change detection (keyed by normalized name so ESPN and nba_api
#  ids can be compared against each other and against old caches)
# ─────────────────────────────────────────────────────────────────

def _player_team_map(rosters):
    mapping = {}
    for team, players in rosters.items():
        if team.startswith("_"):
            continue
        for p in players:
            mapping[_normalize_name(p["name"])] = {"team": team, "name": p["name"], "id": p.get("id")}
    return mapping


def detect_changes(old_rosters, new_rosters):
    """
    Compare old and new rosters and return (trades, added, dropped).
    Only teams present in new_rosters are considered for 'dropped'.
    """
    old_map = _player_team_map(old_rosters)
    new_map = _player_team_map(new_rosters)
    synced_teams = set(new_rosters)

    trades, added, dropped = [], [], []

    for key, new_info in new_map.items():
        old_info = old_map.get(key)
        if old_info is None:
            added.append({"player": new_info["name"], "player_id": new_info["id"], "team": new_info["team"]})
        elif old_info["team"] != new_info["team"]:
            trades.append({
                "player":    new_info["name"],
                "player_id": new_info["id"],
                "from_team": old_info["team"],
                "to_team":   new_info["team"],
            })

    for key, old_info in old_map.items():
        if key not in new_map and old_info["team"] in synced_teams:
            dropped.append({"player": old_info["name"], "player_id": old_info["id"], "team": old_info["team"]})

    return trades, added, dropped


# ─────────────────────────────────────────────────────────────────
#  injuries.json auto-fix
# ─────────────────────────────────────────────────────────────────

def fix_injuries_for_trades(trades):
    """Move a traded player's injury entry in injuries.json to their new team."""
    if not trades or not INJURY_FILE.exists():
        return

    with open(INJURY_FILE) as f:
        injuries = json.load(f)

    changes_made = []
    for trade in trades:
        from_team, to_team, player_name = trade["from_team"], trade["to_team"], trade["player"]
        old_team_injuries = injuries.get(from_team, {})

        matched_key = None
        target = _normalize_name(player_name)
        for key in old_team_injuries:
            if _normalize_name(key) == target:
                matched_key = key
                break
        if matched_key is None:
            last_name = target.split()[-1] if target else ""
            for key in old_team_injuries:
                if last_name and last_name in _normalize_name(key):
                    matched_key = key
                    break

        if matched_key:
            status = old_team_injuries.pop(matched_key)
            injuries.setdefault(to_team, {})[player_name] = status
            changes_made.append(f"    Moved {player_name}: {from_team} -> {to_team} (status: {status})")

    if changes_made:
        print("\n  AUTO-FIXED injuries.json:")
        for c in changes_made:
            print(c)
        with open(INJURY_FILE, "w") as f:
            json.dump(injuries, f, indent=2)
    else:
        print("  No injury entries needed moving.")


# ─────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────

def _print_section(title, rows):
    print(f"\n  {title} ({len(rows)})")
    print("  " + "─" * 50)
    for r in rows:
        print(f"  {r}")


def print_changes(trades, added, dropped, trades_only=False):
    if trades:
        _print_section("TRADES / TEAM CHANGES", [f"{t['player']:<28} {t['from_team']} -> {t['to_team']}" for t in trades])
    elif trades_only:
        print("\n  No trades detected.")

    if trades_only:
        return

    if added:
        _print_section("PLAYERS ADDED TO ROSTERS", [f"{p['player']:<28} signed by {p['team']}" for p in added])
    if dropped:
        _print_section("PLAYERS DROPPED FROM ROSTERS", [f"{p['player']:<28} released by {p['team']}" for p in dropped])
    if not trades and not added and not dropped:
        print("\n  No roster changes detected since last sync.")


def log_changes(trades, added, dropped, failed):
    CACHE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n[{timestamp}]"]
    if not trades and not added and not dropped:
        lines.append("  No changes detected.")
    lines += [f"  TRADE: {t['player']} {t['from_team']} -> {t['to_team']}" for t in trades]
    lines += [f"  ADDED: {p['player']} -> {p['team']}" for p in added]
    lines += [f"  DROPPED: {p['player']} from {p['team']}" for p in dropped]
    lines += [f"  FAILED: {team} ({reason})" for team, reason in failed.items()]
    with open(SYNC_LOG, "a") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync NBA rosters and detect changes")
    parser.add_argument("--team", type=str, help="Sync a single team (e.g. BOS)")
    parser.add_argument("--check", action="store_true", help="Just check if cache is stale")
    parser.add_argument("--trades", action="store_true", help="Show detected trades only")
    parser.add_argument("--source", choices=["auto"] + SOURCE_ORDER, default="auto",
                        help="Roster source (default: auto = espn, then nba)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print(f"  NBA ROSTER SYNC  ({SEASON})")
    print("=" * 55)

    if args.check:
        check_stale()
        return

    old_cache = load_roster_cache()
    old_rosters = {k: v for k, v in old_cache.items() if not k.startswith("_")}

    if args.team:
        team = args.team.upper()
        if team not in ALL_TEAMS:
            print(f"  Unknown team abbreviation: {team}")
            sys.exit(1)
        teams_to_sync = [team]
    else:
        teams_to_sync = ALL_TEAMS

    check_stale(warn_only=True)
    print(f"  Syncing {len(teams_to_sync)} team(s) via '{args.source}'...\n")

    new_rosters, failed = fetch_all_rosters(teams_to_sync, source=args.source)

    if not new_rosters:
        print("\n  No rosters fetched from any source. Check your connection.")
        sys.exit(1)

    if failed:
        print(f"\n  {len(failed)} team(s) failed; keeping their previous roster in cache:")
        for team, reason in failed.items():
            print(f"    {team}: {reason}")

    if old_rosters:
        # Pass every old team so a player arriving from an unsynced team still registers as a trade
        trades, added, dropped = detect_changes(old_rosters, new_rosters)
    else:
        print("\n  First sync — no previous roster to compare against.")
        trades, added, dropped = [], [], []

    print_changes(trades, added, dropped, trades_only=args.trades)

    if trades:
        print("\n  Checking injuries.json for affected players...")
        fix_injuries_for_trades(trades)

    merged = {**old_rosters, **new_rosters}
    save_roster_cache(merged)
    log_changes(trades, added, dropped, failed)

    print(f"\n  Synced {len(new_rosters)}/{len(teams_to_sync)} team(s). Cache updated: {ROSTER_CACHE}")
    print(f"  Change log: {SYNC_LOG}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
