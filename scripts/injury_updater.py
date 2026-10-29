import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import ssl
import time
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

INJURY_FILE = Path("injuries.json")

ESPN_URL = "https://www.espn.com/nba/injuries"
ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# ESPN sometimes uses different abbreviations than this project
ESPN_ABBR_TO_NBA = {
    "GS": "GSW",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "NY": "NYK",
    "NO": "NOP",
    "PHO": "PHX",
    "BK": "BKN",
}

# ESPN full team name -> NBA abbreviation
TEAM_NAME_TO_ABBR = {
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

# ESPN status -> injuries.json status (keys are alphanumerics-only, lowercased)
STATUS_MAP = {
    "out": "out",
    "injuredreserve": "out",
    "ir": "out",
    "suspended": "out",
    "doubtful": "out",
    "daytoday": "questionable",
    "gtd": "questionable",
    "gametimedecision": "questionable",
    "probable": "questionable",
    "questionable": "questionable",
}

# Players to always skip — refs, coaches, two-way fringe
SKIP_KEYWORDS = ["reconditioning", "g league", "two-way", "not with team"]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize_status(raw):
    """Map an ESPN status string to out / questionable, or None if unknown."""
    if not raw:
        return None
    key = re.sub(r"[^a-z]+", "", str(raw).lower())
    return STATUS_MAP.get(key)


def _team_abbr(display_name=None, espn_abbr=None):
    if display_name:
        for team_name, abbr in TEAM_NAME_TO_ABBR.items():
            if team_name.lower() == str(display_name).lower():
                return abbr
    if espn_abbr:
        espn_abbr = str(espn_abbr).upper()
        return ESPN_ABBR_TO_NBA.get(espn_abbr, espn_abbr)
    return None


def _should_skip(text):
    lowered = (text or "").lower()
    return any(kw in lowered for kw in SKIP_KEYWORDS)


def _http_get(url, accept=None):
    headers = dict(_REQUEST_HEADERS)
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            if attempt < 2:
                print(f"  Retry {attempt + 1}/3...")
                time.sleep(3)
    raise last_err


def fetch_espn_injury_json():
    """Fetch ESPN's structured injury feed (avoids the HTML WAF challenge)."""
    raw = _http_get(ESPN_API_URL, accept="application/json")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or "injuries" not in data:
        raise ValueError("Unexpected ESPN injury API payload")
    return data


def parse_injuries_api(data):
    """
    Parse ESPN site API JSON into {team_abbr: {player: status}}.

    Team assignment uses the table/section team (displayName), not athlete.team,
    because traded players can still be listed under their old club.
    """
    injuries = {}
    for team in data.get("injuries") or []:
        abbr = _team_abbr(display_name=team.get("displayName"))
        if not abbr:
            continue
        injuries.setdefault(abbr, {})
        for entry in team.get("injuries") or []:
            athlete = entry.get("athlete") or {}
            player_name = (athlete.get("displayName") or "").strip()
            if not player_name:
                continue

            # Skip only on the player name — comments often mention "G League"/"two-way"
            # in passing and would drop real injuries.
            if _should_skip(player_name):
                continue

            status = _normalize_status(entry.get("status"))
            if status is None:
                type_info = entry.get("type") or {}
                status = _normalize_status(type_info.get("description") or type_info.get("name"))
            if status is None:
                continue

            injuries[abbr][player_name] = status
    return injuries


def fetch_espn_injuries():
    """
    Fetch ESPN injury page and return raw HTML.
    Uses urllib to avoid extra dependencies.
    """
    raw = _http_get(ESPN_URL, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    return raw.decode("utf-8", errors="replace")


def parse_injuries(html):
    """
    Parse ESPN injury HTML into a dict of {team_abbr: {player: status}}.

    ESPN structures the page as team headers followed by tables.
    We extract team sections then parse each player row.
    """
    injuries = {}
    current_team = None
    current_abbr = None

    # Split into lines for processing
    lines = html.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # ── Detect team headers ──
        # ESPN team headers appear as the team name in heading tags
        for team_name, abbr in TEAM_NAME_TO_ABBR.items():
            if team_name in line and ("<h" in line.lower() or team_name == line):
                current_team = team_name
                current_abbr = abbr
                if abbr not in injuries:
                    injuries[abbr] = {}
                break

        # ── Detect player rows ──
        # Player rows contain status keywords and player names
        if current_abbr and ("Out" in line or "Day-To-Day" in line or "Probable" in line or "Questionable" in line or "Doubtful" in line):
            # Try to extract player name and status from the line
            player_name = None
            status = None
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", line):
                mapped = _normalize_status(token)
                if mapped:
                    status = mapped
                    break

            if status and current_abbr:
                # Extract player name — appears before the status in ESPN rows
                # Clean HTML tags first
                clean_line = re.sub(r"<[^>]+>", " ", line)
                clean_line = re.sub(r"\s+", " ", clean_line).strip()

                # Skip fringe/non-meaningful entries
                skip = any(kw in clean_line.lower() for kw in SKIP_KEYWORDS)
                if not skip and len(clean_line) > 3:
                    # Player name is usually the first meaningful text
                    # before position abbreviations (G, F, C, G/F, etc.)
                    name_match = re.match(
                        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z'.-]+(?:\s+(?:Jr\.|Sr\.|III|II|IV))?)?)",
                        clean_line
                    )
                    if name_match:
                        player_name = name_match.group(1).strip()
                        # Filter out team names accidentally matched
                        if player_name and player_name not in TEAM_NAME_TO_ABBR and len(player_name) > 4:
                            injuries[current_abbr][player_name] = status

        i += 1

    return injuries


def parse_injuries_structured(html):
    """
    More robust parser using ESPN's structured data patterns.
    Falls back to regex-based extraction from the page content.
    """
    injuries = {}

    # Remove script/style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)

    # Find all team sections
    # ESPN wraps each team in a section with the team name
    team_pattern = re.compile(
        r'(?:' + '|'.join(re.escape(t) for t in TEAM_NAME_TO_ABBR.keys()) + r')',
        re.IGNORECASE
    )

    # Split HTML around team names to get sections
    parts = team_pattern.split(html)
    team_matches = team_pattern.findall(html)

    current_injuries = {}

    for idx, team_name_raw in enumerate(team_matches):
        # Normalize team name
        team_name = None
        abbr = None
        for tn, ab in TEAM_NAME_TO_ABBR.items():
            if tn.lower() == team_name_raw.lower():
                team_name = tn
                abbr = ab
                break

        if not abbr:
            continue

        if abbr not in current_injuries:
            current_injuries[abbr] = {}

        if idx + 1 >= len(parts):
            continue

        section = parts[idx + 1]

        # Extract player rows from this section
        # Each row contains: name, position, return date, status, comment
        rows = re.findall(
            r'<tr[^>]*>(.*?)</tr>',
            section,
            re.DOTALL
        )

        for row in rows:
            # Clean tags from row
            clean = re.sub(r"<[^>]+>", " ", row)
            clean = re.sub(r"&amp;", "&", clean)
            clean = re.sub(r"&[a-z]+;", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

            if not clean or len(clean) < 5:
                continue

            # Skip header rows
            if any(h in clean for h in ["NAME", "POS", "STATUS", "COMMENT", "EST."]):
                continue

            # Skip fringe entries
            if any(kw in clean.lower() for kw in SKIP_KEYWORDS):
                continue

            status = None
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", clean):
                mapped = _normalize_status(token)
                if mapped:
                    status = mapped
                    break

            if not status:
                continue

            # Extract player name — first capitalized word sequence
            name_match = re.match(
                r"([A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){1,3})",
                clean
            )
            if not name_match:
                continue

            player_name = name_match.group(1).strip()

            # Validate — filter out positions, team names, status words
            invalid = ["Out", "Day", "Probable", "Questionable", "Doubtful",
                      "Guard", "Forward", "Center", "Mar", "Feb", "Jan",
                      "Apr", "Oct", "Nov", "Dec"] + list(TEAM_NAME_TO_ABBR.keys())
            if any(player_name.startswith(inv) for inv in invalid):
                continue

            if len(player_name) < 5 or len(player_name.split()) < 2:
                continue

            current_injuries[abbr][player_name] = status

    return current_injuries


def load_existing_injuries():
    """Load existing injuries.json."""
    if not INJURY_FILE.exists():
        return {}
    with open(INJURY_FILE) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_injuries(injuries):
    """Save updated injuries to injuries.json."""
    output = {
        "_instructions": "Auto-updated from ESPN. Set status to: 'out', 'questionable', or 'available'.",
        "_last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # Ensure all 30 teams are present even if no injuries
    all_teams = list(dict.fromkeys(TEAM_NAME_TO_ABBR.values()))
    for team in all_teams:
        output[team] = injuries.get(team, {})

    with open(INJURY_FILE, "w") as f:
        json.dump(output, f, indent=2)


def print_injuries(injuries):
    """Print current injury report."""
    print(f"\n  CURRENT INJURY REPORT")
    print(f"  {'─'*55}")
    total = 0
    for team in sorted(injuries.keys()):
        players = injuries[team]
        if players:
            print(f"\n  {team}:")
            for player, status in players.items():
                print(f"    {player:<28} {status}")
                total += 1
    print(f"\n  {'─'*55}")
    print(f"  Total injuries: {total}\n")


def update(show_only=False, verbose=True):
    """
    Main update function. Fetches ESPN and updates injuries.json.
    Returns the updated injuries dict.
    """
    if verbose:
        print("  Fetching injury report from ESPN...")

    injuries = {}

    try:
        api_data = fetch_espn_injury_json()
        injuries = parse_injuries_api(api_data)
    except Exception as e:
        if verbose:
            print(f"  ESPN injury API failed ({e}); trying HTML page...")

    if not injuries or sum(len(v) for v in injuries.values()) < 5:
        try:
            html = fetch_espn_injuries()
        except Exception as e:
            print(f"  Failed to fetch ESPN injuries: {e}")
            print("  Using existing injuries.json")
            return load_existing_injuries()

        if "AwsWafIntegration" in html or "challenge.js" in html:
            print("  Warning: ESPN HTML is a WAF challenge page — keeping existing file")
            return load_existing_injuries()

        injuries = parse_injuries_structured(html)
        if not injuries or sum(len(v) for v in injuries.values()) < 5:
            injuries = parse_injuries(html)

    if not injuries or sum(len(v) for v in injuries.values()) < 5:
        print("  Warning: Could not parse ESPN injuries — keeping existing file")
        return load_existing_injuries()

    total = sum(len(v) for v in injuries.values())

    if show_only:
        print_injuries(injuries)
        return injuries

    if not show_only:
        save_injuries(injuries)
        if verbose:
            print(f"  Updated injuries.json — {total} players listed")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return injuries


def main():
    parser = argparse.ArgumentParser(description="Auto-update injuries.json from ESPN")
    parser.add_argument("--show", action="store_true", help="Show injuries without saving")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    print("\n  NBA INJURY UPDATER")
    print("  " + "─" * 40)
    update(show_only=args.show, verbose=not args.quiet)


if __name__ == "__main__":
    main()