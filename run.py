"""
run.py — Convenience runner for NBA AI Agent Game Predictor

Usage:
    python run.py schedule              # fetch today's games + predict
    python run.py results               # auto-fill last night's results
    python run.py results --date 2026-03-04
    python run.py summary               # show running ROI
    python run.py preview               # show today's schedule only
    python run.py rankings              # Elo power rankings
    python run.py train                 # train main models
    python run.py train-nn              # train player value NN
    python run.py train-nn --team OKC   # show one team's tiers
    python run.py train-nn --tier Star  # show all stars league-wide
    python run.py roster                # sync all 30 rosters
    python run.py roster --team LAL     # sync one team
    python run.py roster --check        # check if cache is stale
    python run.py injuries              # update injuries.json from ESPN
    python run.py backtest              # A/B/C injury model comparison
    python run.py tracker --predict BOS LAL
    python run.py tracker --summary
    python run.py main --home BOS --away LAL
    python run.py main --games OKC:BOS LAL:GSW MIA:NYK
    python run.py main --backtest 2024-25
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# ── Script mapping ────────────────────────────────────────────────
SCRIPTS = {
    "schedule":  ROOT / "scripts/schedule.py",
    "main":      ROOT / "scripts/main.py",
    "train":     ROOT / "scripts/train.py",
    "train-nn":  ROOT / "scripts/train_player_nn.py",
    "roster":    ROOT / "scripts/roster_sync.py",
    "injuries":  ROOT / "scripts/injury_updater.py",
    "tracker":   ROOT / "scripts/prediction_tracker.py",
    "backtest":  ROOT / "scripts/backtest_injury.py",
}

# ── Shortcuts (expand to script + args) ──────────────────────────
SHORTCUTS = {
    "results":   ("schedule",  ["--results"]),
    "summary":   ("schedule",  ["--summary"]),
    "preview":   ("schedule",  ["--preview"]),
    "rankings":  ("main",      ["--rankings"]),
}

# ── Help text ─────────────────────────────────────────────────────
HELP = """
  NBA AI AGENT GAME PREDICTOR
  ─────────────────────────────────────────────────────

  DAILY WORKFLOW
    schedule          Fetch today's games + predict all
    results           Auto-fill last night's results
    summary           Show running ROI and accuracy
    preview           Show today's schedule without predicting

  PREDICTIONS
    main              Predict specific games
      --home BOS --away LAL
      --games OKC:BOS LAL:GSW MIA:NYK
      --rankings
      --backtest 2024-25

  TRAINING
    train             Train main LR + XGBoost models
    train-nn          Train player value neural network
      --team OKC      Show one team's player tiers
      --tier Star     Show all players of a tier
      --force         Re-fetch player data from API

  MAINTENANCE
    roster            Sync all 30 rosters from NBA API
      --team LAL      Sync one team only
      --check         Check if roster cache is stale
      --trades        Show detected trades only
    injuries          Update injuries.json from ESPN

  ANALYSIS
    backtest          Compare Model A / B / C injury systems
      --season 2023-24
      --model C
      --min-edge 0.04
      
    tracker           Manual prediction logger
      --predict BOS LAL
      --result BOS LAL 1
      --summary
      --pending

  ─────────────────────────────────────────────────────
  Examples:
    python run.py schedule
    python run.py results --date 2026-03-04
    python run.py main --home OKC --away BOS
    python run.py train-nn --tier Franchise
    python run.py clv --summary
  ─────────────────────────────────────────────────────
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        return

    cmd = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    # Expand shortcuts
    if cmd in SHORTCUTS:
        script_key, shortcut_args = SHORTCUTS[cmd]
        extra_args = shortcut_args + extra_args
        cmd = script_key

    script = SCRIPTS.get(cmd)
    if not script:
        print(f"\n  Unknown command: '{cmd}'")
        print(f"  Run 'python run.py help' to see all commands.\n")
        sys.exit(1)

    if not script.exists():
        print(f"\n  Script not found: {script}")
        print(f"  Make sure you have run the migration and all files are in scripts/\n")
        sys.exit(1)

    # Run the script
    result = subprocess.run(
        [sys.executable, str(script)] + extra_args,
        cwd=str(ROOT)
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()