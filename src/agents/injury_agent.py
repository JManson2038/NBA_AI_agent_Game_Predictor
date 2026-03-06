
import json
import numpy as np
from pathlib import Path

from src.models.player_value_nn import TIERS, TIER_ELO_RANGE, score_to_tier
import config


CACHE_DIR = Path("cache")
INJURY_FILE = Path("injuries.json")

DEFAULT_INJURIES = {
    "_instructions": "Set status to: 'out', 'questionable', or 'available'",
}

# Questionable multiplier range (random sample within for realism)
QUESTIONABLE_DISCOUNT = (0.4, 0.6)

# Hard cap: no team loses more than this from injuries
ELO_CAP = 180

# Status multipliers
STATUS_MULT = {
    "out":          1.0,
    "questionable": None,   # Sampled from QUESTIONABLE_DISCOUNT
    "available":    0.0,
}


class InjuryAgent:
    """
    Phase 3 calibrated injury agent.

    Tier classification uses objective metrics only:
      - Minutes per game (MPG)
      - Usage rate (USG%)
      - Net rating on/off
      - BPM proxy

    No manual tagging. No flat universal penalties.
    """

    def __init__(self, season="2025-26"):
        self.season = season
        self.injuries = self._load_injury_file()
        self._player_cache = {}

    # ── Injury file ──────────────────────────────────────────────

    def _load_injury_file(self):
        if not INJURY_FILE.exists():
            with open(INJURY_FILE, "w") as f:
                json.dump(DEFAULT_INJURIES, f, indent=2)
            print(f"[InjuryAgent] Created {INJURY_FILE}")
        with open(INJURY_FILE) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def reload_injuries(self):
        self.injuries = self._load_injury_file()
        print("[InjuryAgent] Reloaded.")

    def set_injury(self, team_abbr, player_name, status):
        assert status in ("out", "questionable", "available")
        team_abbr = team_abbr.upper()
        if team_abbr not in self.injuries:
            self.injuries[team_abbr] = {}
        self.injuries[team_abbr][player_name] = status
        save_data = {"_instructions": DEFAULT_INJURIES.get("_instructions", "")}
        save_data.update(self.injuries)
        with open(INJURY_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"[InjuryAgent] {player_name} ({team_abbr}) -> '{status}'")

    # ── Player stats lookup ──────────────────────────────────────

    def _get_team_players(self, team_abbr):
        if team_abbr in self._player_cache:
            return self._player_cache[team_abbr]

        cache_file = CACHE_DIR / f"player_stats_{self.season}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                all_players = json.load(f)
            team_players = [
                p for p in all_players
                if p.get("TEAM_ABBREVIATION", "") == team_abbr
            ]
            self._player_cache[team_abbr] = team_players
            return team_players
        return []

    def _find_player(self, player_name, team_abbr):
        players = self._get_team_players(team_abbr)
        # Exact match
        for p in players:
            name = p.get("PLAYER_NAME", p.get("PLAYER", ""))
            if name.lower() == player_name.lower():
                return p
        # Last name match
        last = player_name.split()[-1].lower()
        for p in players:
            name = p.get("PLAYER_NAME", p.get("PLAYER", ""))
            if last in name.lower():
                return p
        return None

    # ── Phase 3: Objective tier classification ───────────────────

    def _classify_tier(self, record):
        """
        Classify player tier using the PlayerValueNN score.
        Uses the same TIERS and score_to_tier as train_player_nn.py
        so display and penalty tiers are always in sync.
        """
        try:
            from src.models.player_value_nn import PlayerValueNN, generate_importance_label
            if not hasattr(self, '_nn'):
                self._nn = PlayerValueNN()
                if not self._nn.load():
                    self._nn = None

            if self._nn:
                score = self._nn.predict_player(record)
            else:
                score = generate_importance_label(record)
        except Exception:
            score = generate_importance_label(record)

        tier = score_to_tier(score)
        lo, hi = TIER_ELO_RANGE.get(tier, (10, 25))
        elo_penalty = round(lo + score * (hi - lo), 1)
        return tier, elo_penalty

    def _get_questionable_mult(self):
        """Sample questionable discount from range for realism."""
        lo, hi = QUESTIONABLE_DISCOUNT
        return round(np.random.uniform(lo, hi), 3)

    # ── Core computation ─────────────────────────────────────────

    def compute_team_impact(self, team_abbr):
        """
        Compute total Elo penalty for a team based on injuries.

        Returns:
            total_penalty: float (Elo points to subtract)
            details:       list of per-player dicts
        """
        team_abbr = team_abbr.upper()
        team_injuries = self.injuries.get(team_abbr, {})

        if not team_injuries:
            return 0.0, []

        total_penalty = 0.0
        details = []

        for player_name, status in team_injuries.items():
            if status == "available":
                continue

            # Get status multiplier
            if status == "questionable":
                mult = self._get_questionable_mult()
            else:
                mult = STATUS_MULT.get(status, 1.0)

            # Look up player stats
            record = self._find_player(player_name, team_abbr)

            if record:
                tier, base_penalty = self._classify_tier(record)
                mpg = round(record.get("MIN", 0), 1)
                ppg = round(record.get("PTS", 0), 1)
                usg = round((record.get("USG_PCT", 0) or 0) * 100, 1)
                source = "stats"
            else:
                # Unknown player — default to Starter tier
                tier = "Starter"
                base_penalty = 35.0
                mpg, ppg, usg = 0.0, 0.0, 0.0
                source = "default"

            penalty = round(base_penalty * mult, 1)
            total_penalty += penalty

            details.append({
                "player":       player_name,
                "status":       status,
                "tier":         tier,
                "base_penalty": base_penalty,
                "multiplier":   mult,
                "elo_penalty":  penalty,
                "mpg":          mpg,
                "ppg":          ppg,
                "usg":          usg,
                "source":       source,
            })

        # Phase 3: Hard cap at ELO_CAP
        if total_penalty > ELO_CAP:
            scale = ELO_CAP / total_penalty
            for d in details:
                d["elo_penalty"] = round(d["elo_penalty"] * scale, 1)
            total_penalty = ELO_CAP
            details.append({"player": "-- CAP APPLIED --", "elo_penalty": 0})

        details = [d for d in details if d.get("player") != "-- CAP APPLIED --"]
        details.sort(key=lambda x: -x["elo_penalty"])

        return round(total_penalty, 1), details

    def get_adjusted_elo(self, team_abbr, base_elo):
        """Apply calibrated injury penalty to Elo."""
        penalty, details = self.compute_team_impact(team_abbr)
        adjusted = round(base_elo - penalty, 1)
        return adjusted, penalty, details

    def explain(self, home_abbr, away_abbr):
        """Print injury report for both teams."""
        print("\n  INJURY REPORT  (Phase 3 Calibrated)")
        print("  " + "─" * 65)

        for team, label in [(home_abbr, "HOME"), (away_abbr, "AWAY")]:
            penalty, details = self.compute_team_impact(team)

            if not details:
                print(f"  {label} {team}: No injuries")
            else:
                capped = " [CAPPED]" if penalty >= ELO_CAP else ""
                print(f"  {label} {team}: Total Elo penalty = -{penalty}{capped}")
                print(f"  {'Player':<26} {'Status':<12} {'Tier':<10} {'MPG':>5} {'USG%':>5}  {'Elo':>6}")
                print("  " + "─" * 65)
                for d in details:
                    print(
                        f"  {d['player']:<26} {d['status']:<12} {d['tier']:<10} "
                        f"{d['mpg']:>5.1f} {d['usg']:>5.1f}  -{d['elo_penalty']:>5.1f}"
                    )

        print("  " + "─" * 65)