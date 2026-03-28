
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import CommonTeamRoster

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

CACHE_DIR = Path("cache")
INJURY_FILE = Path("injuries.json")
ROSTER_CACHE = CACHE_DIR / "rosters.json"
SYNC_LOG = CACHE_DIR / "roster_sync.log"
SEASON = "2025-26"
STALE_DAYS = 3  # Warn if cache older than this

ALL_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA", "WAS",
]



#  Cache helpers

def load_roster_cache():
    """Load existing roster cache or return empty dict."""
    if ROSTER_CACHE.exists():
        with open(ROSTER_CACHE) as f:
            return json.load(f)
    return {}


def save_roster_cache(data):
    """Save roster cache with timestamp."""
    data["_last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(ROSTER_CACHE, "w") as f:
        json.dump(data, f, indent=2)


def check_stale(warn_only=False):
    """Check if roster cache is stale and warn."""
    if not ROSTER_CACHE.exists():
        print("  No roster cache found. Run: python roster_sync.py")
        return True

    data = load_roster_cache()
    last_updated = data.get("_last_updated")

    if not last_updated:
        print("  Roster cache has no timestamp. Run: python roster_sync.py")
        return True

    last_dt = datetime.fromisoformat(last_updated)
    now = datetime.now(timezone.utc)
    age_days = (now - last_dt).days
    age_hours = int((now - last_dt).total_seconds() / 3600)

    if age_days >= STALE_DAYS:
        print(f"  WARNING: Roster cache is {age_days} days old (last updated: {last_updated[:10]})")
        print(f"  Run: python roster_sync.py  to get current rosters")
        return True
    else:
        if not warn_only:
            print(f"  Roster cache is fresh ({age_hours}h old, updated {last_updated[:10]})")
        return False



#  Roster fetching

def fetch_team_roster(team_id, team_abbr, retries=3, delay=5):
    """Fetch current roster for one team from nba_api with retries."""
    for attempt in range(retries):
        try:
            roster = CommonTeamRoster(
                team_id=team_id,
                season=SEASON,
                headers=HEADERS,
                timeout=60,
            )
            time.sleep(0.6)
            df = roster.get_data_frames()[0]

            players = []
            for _, row in df.iterrows():
                players.append({
                    "id": int(row["PLAYER_ID"]),
                    "name": str(row["PLAYER"]),
                    "num": str(row.get("NUM", "")),
                    "position": str(row.get("POSITION", "")),
                })

            return players

        except Exception as e:
            if attempt < retries - 1:
                print(f"timeout, retrying in {delay}s...", end=" ")
                time.sleep(delay)
            else:
                print(f"failed after {retries} attempts: {e}")
                return None


def fetch_all_rosters(teams_to_sync):
    """Fetch rosters for a list of team abbreviations."""
    all_nba_teams = nba_teams.get_teams()
    abbr_to_id = {t["abbreviation"]: t["id"] for t in all_nba_teams}

    rosters = {}
    for i, abbr in enumerate(teams_to_sync):
        team_id = abbr_to_id.get(abbr)
        if not team_id:
            print(f"  Unknown team abbreviation: {abbr}")
            continue

        print(f"  [{i+1}/{len(teams_to_sync)}] Fetching {abbr}...", end=" ")
        players = fetch_team_roster(team_id, abbr)

        if players is not None:
            rosters[abbr] = players
            print(f"{len(players)} players")
        else:
            print("failed")

    return rosters



#  Change detection

def detect_changes(old_rosters, new_rosters):
    """
    Compare old and new rosters to find:
    - Players who switched teams (trades)
    - Players added to a roster
    - Players dropped from a roster
    """
    # Build player -> team maps
    old_player_team = {}
    for team, players in old_rosters.items():
        if team.startswith("_"):
            continue
        for p in players:
            old_player_team[p["id"]] = {"team": team, "name": p["name"]}

    new_player_team = {}
    for team, players in new_rosters.items():
        for p in players:
            new_player_team[p["id"]] = {"team": team, "name": p["name"]}

    trades = []
    added = []
    dropped = []

    # Check every player in new rosters
    for pid, new_info in new_player_team.items():
        if pid in old_player_team:
            old_info = old_player_team[pid]
            if old_info["team"] != new_info["team"]:
                trades.append({
                    "player": new_info["name"],
                    "player_id": pid,
                    "from_team": old_info["team"],
                    "to_team": new_info["team"],
                })
        else:
            added.append({
                "player": new_info["name"],
                "player_id": pid,
                "team": new_info["team"],
            })

    # Check for dropped players
    for pid, old_info in old_player_team.items():
        if pid not in new_player_team:
            dropped.append({
                "player": old_info["name"],
                "player_id": pid,
                "team": old_info["team"],
            })

    return trades, added, dropped



#  injuries.json auto-fix

def fix_injuries_for_trades(trades):
    """
    When a player switches teams in a trade, move their
    injury entry in injuries.json to the new team.
    """
    if not trades:
        return

    if not INJURY_FILE.exists():
        return

    with open(INJURY_FILE) as f:
        injuries = json.load(f)

    changes_made = []

    for trade in trades:
        player_name = trade["player"]
        from_team = trade["from_team"]
        to_team = trade["to_team"]

        # Check if this player has an injury entry on their old team
        old_team_injuries = injuries.get(from_team, {})

        # Try exact match first, then partial (last name)
        matched_key = None
        if player_name in old_team_injuries:
            matched_key = player_name
        else:
            last_name = player_name.split()[-1].lower()
            for key in old_team_injuries:
                if last_name in key.lower():
                    matched_key = key
                    break

        if matched_key:
            status = old_team_injuries[matched_key]

            # Remove from old team
            del injuries[from_team][matched_key]

            # Add to new team
            if to_team not in injuries:
                injuries[to_team] = {}
            injuries[to_team][player_name] = status

            changes_made.append(
                f"    Moved {player_name}: {from_team} -> {to_team} (status: {status})"
            )

    if changes_made:
        print("\n  AUTO-FIXED injuries.json:")
        for c in changes_made:
            print(c)

        # Save updated injuries
        with open(INJURY_FILE, "w") as f:
            json.dump(injuries, f, indent=2)
    else:
        print("  No injury entries needed moving.")



#  Reporting

def print_changes(trades, added, dropped):
    #Print a clean summary of all roster changes

    if not trades and not added and not dropped:
        print("\n  No roster changes detected since last sync.")
        return

    if trades:
        print(f"\n  TRADES / TEAM CHANGES ({len(trades)})")
        print("  " + "─" * 50)
        for t in trades:
            print(f"  {t['player']:<28} {t['from_team']} -> {t['to_team']}")

    if added:
        print(f"\n  PLAYERS ADDED TO ROSTERS ({len(added)})")
        print("  " + "─" * 50)
        for p in added:
            print(f"  {p['player']:<28} signed by {p['team']}")

    if dropped:
        print(f"\n  PLAYERS DROPPED FROM ROSTERS ({len(dropped)})")
        print("  " + "─" * 50)
        for p in dropped:
            print(f"  {p['player']:<28} released by {p['team']}")


def log_changes(trades, added, dropped):
    #Append changes to a sync log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n[{timestamp}]"]

    if not trades and not added and not dropped:
        lines.append("  No changes detected.")
    for t in trades:
        lines.append(f"  TRADE: {t['player']} {t['from_team']} -> {t['to_team']}")
    for p in added:
        lines.append(f"  ADDED: {p['player']} -> {p['team']}")
    for p in dropped:
        lines.append(f"  DROPPED: {p['player']} from {p['team']}")

    with open(SYNC_LOG, "a") as f:
        f.write("\n".join(lines))



#  Main

def main():
    parser = argparse.ArgumentParser(description="Sync NBA rosters and detect changes")
    parser.add_argument("--team", type=str, help="Sync a single team (e.g. BOS)")
    parser.add_argument("--check", action="store_true", help="Just check if cache is stale")
    parser.add_argument("--trades", action="store_true", help="Show detected trades only")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  NBA ROSTER SYNC")
    print("=" * 55)

    # ── Just check staleness ──
    if args.check:
        check_stale()
        return

    # ── Load existing cache ──
    old_cache = load_roster_cache()
    old_rosters = {k: v for k, v in old_cache.items() if not k.startswith("_")}

    # ── Decide which teams to sync ──
    if args.team:
        teams_to_sync = [args.team.upper()]
    else:
        teams_to_sync = ALL_TEAMS

    # ── Check staleness first ──
    is_stale = check_stale(warn_only=True)
    if is_stale:
        print(f"  Syncing {len(teams_to_sync)} team(s)...\n")

    # ── Fetch fresh rosters ──
    print(f"  Fetching rosters from nba_api...")
    new_rosters = fetch_all_rosters(teams_to_sync)

    if not new_rosters:
        print("  No rosters fetched. Check your connection.")
        return

    # ── Merge with existing cache (keep teams not in this sync) ──
    merged = {**old_rosters, **new_rosters}

    # ── Detect changes ──
    if old_rosters:
        # Only compare teams we actually synced
        old_subset = {t: old_rosters[t] for t in teams_to_sync if t in old_rosters}
        trades, added, dropped = detect_changes(old_subset, new_rosters)
    else:
        print("\n  First sync — no previous roster to compare against.")
        trades, added, dropped = [], [], []

    # ── Show results ──
    if args.trades:
        if trades:
            print(f"\n  TRADES / TEAM CHANGES ({len(trades)})")
            print("  " + "─" * 50)
            for t in trades:
                print(f"  {t['player']:<28} {t['from_team']} -> {t['to_team']}")
        else:
            print("\n  No trades detected.")
    else:
        print_changes(trades, added, dropped)

    # ── Auto-fix injuries.json for trades ──
    if trades:
        print("\n  Checking injuries.json for affected players...")
        fix_injuries_for_trades(trades)

    # ── Save updated cache ──
    save_roster_cache(merged)
    log_changes(trades, added, dropped)

    print(f"\n  Sync complete. Cache updated: cache/rosters.json")
    print(f"  Change log: cache/roster_sync.log")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()