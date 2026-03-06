"""
train_player_nn.py
Train the PlayerValueNN and print ranked player tiers for all 30 teams.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from pathlib import Path
from src.models.player_value_nn import PlayerValueNN, PlayerDataCollector

ALL_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA", "WAS",
]


def load_player_cache(season="2024-25"):
    """Load all player stats from cache."""
    cache_file = Path(f"cache/player_stats_{season}.json")
    if not cache_file.exists():
        print(f"No player stats cache found for {season}.")
        print("Run: python train_player_nn.py  (without --team) to fetch and train first.")
        return []
    with open(cache_file) as f:
        return json.load(f)


def get_team_players(all_players, team_abbr):
    """Filter players by team abbreviation."""
    return [
        p for p in all_players
        if p.get("TEAM_ABBREVIATION", "") == team_abbr
    ]


def show_all_teams(nn, all_players):
    """Print rankings for all 30 teams."""
    for team in ALL_TEAMS:
        team_players = get_team_players(all_players, team)
        if team_players:
            nn.print_team_rankings(team, team_players)
        else:
            print(f"\n  {team}: No player data found in cache.")


def show_one_team(nn, all_players, team_abbr):
    """Print rankings for a single team."""
    team_abbr = team_abbr.upper()
    team_players = get_team_players(all_players, team_abbr)
    if not team_players:
        print(f"No players found for {team_abbr} in cache.")
        return
    nn.print_team_rankings(team_abbr, team_players)


def show_by_tier(nn, all_players, tier_filter):
    """Print all players league-wide matching a specific tier."""
    from player_value_nn import score_to_tier

    print(f"\n  ALL PLAYERS — Tier: {tier_filter}")
    print("  " + "─" * 70)
    print(f"  {'Team':<6} {'Player':<26} {'Tier':<14} {'Score':>6}  {'MPG':>5}  {'PPG':>5}  {'USG%':>5}")
    print("  " + "─" * 70)

    matches = []
    for p in all_players:
        try:
            score = nn.predict_player(p)
            tier = score_to_tier(score)
            if tier.lower() == tier_filter.lower():
                matches.append({
                    "team": p.get("TEAM_ABBREVIATION", "???"),
                    "player": p.get("PLAYER_NAME", p.get("PLAYER", "Unknown")),
                    "tier": tier,
                    "score": score,
                    "mpg": round(p.get("MIN", 0), 1),
                    "ppg": round(p.get("PTS", 0), 1),
                    "usg": round(p.get("USG_PCT", 0) * 100, 1),
                })
        except Exception:
            continue

    # Sort by score descending
    matches.sort(key=lambda x: -x["score"])

    for p in matches:
        print(
            f"  {p['team']:<6} {p['player']:<26} {p['tier']:<14} "
            f"{p['score']:>6.3f}  {p['mpg']:>5.1f}  {p['ppg']:>5.1f}  {p['usg']:>5.1f}"
        )

    print("  " + "─" * 70)
    print(f"  Total: {len(matches)} players\n")


def main():
    parser = argparse.ArgumentParser(description="Train and view NBA player value tiers")
    parser.add_argument("--force", action="store_true", help="Re-fetch all player data from API")
    parser.add_argument("--team", type=str, help="Show rankings for one team (e.g. BOS)")
    parser.add_argument("--tier", type=str, help="Show all players of one tier league-wide (e.g. Franchise)")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    nn = PlayerValueNN()

    # ── Team or tier view (no retraining needed) ──
    if args.team or args.tier:
        if not nn.load():
            print("No trained model found. Run without --team or --tier first to train.")
            return
        all_players = load_player_cache()
        if not all_players:
            return
        if args.team:
            show_one_team(nn, all_players, args.team)
        elif args.tier:
            show_by_tier(nn, all_players, args.tier)
        return

    # ── Full training run ──
    nn.train(force_fetch=args.force, epochs=args.epochs)

    all_players = load_player_cache()
    if not all_players:
        return

    show_all_teams(nn, all_players)


if __name__ == "__main__":
    main()