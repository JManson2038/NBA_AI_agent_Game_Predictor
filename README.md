# NBA AI Agent Game Predictor

A multi-agent machine learning system that predicts NBA game outcomes daily. Built with Elo ratings, rolling team statistics, matchup analysis, a neural network for player importance scoring, and automated injury tracking.

---

## How It Works

```
NBA API
  └── Data Agent          fetches game logs, engineers rolling features
        └── Team Strength Agent    builds Elo ratings (margin-of-victory adjusted)
              └── Matchup Agent         detects pace, shooting, rebounding mismatches
                    └── Prediction Agent      LR + XGBoost ensemble
                          └── Confidence Agent      scores prediction reliability
                                └── Injury Agent          calibrated Elo penalties via PlayerValueNN
                                      └── Orchestrator          final prediction with explanation
```

---

> Predictions are logged daily with full injury context, confidence tier, and spread. Results are auto-filled each evening from the NBA scoreboard.

---

## Daily Workflow

```bash
# Morning — fetch schedule, update injuries, predict all games
python run.py schedule

# Evening — auto-fill results from NBA scoreboard
python run.py results

# Check running ROI
python run.py summary
```

---

## Sample Output

```
  ══════════════════════════════════════════════════════════
  NBA SCHEDULE — Wednesday, March 4, 2026
  ══════════════════════════════════════════════════════════

  BKN  @ MIA   Pick: MIA   71.5%  Spread: +6.5  Conf: High (0.765)   [INJ: -119H/-35A]
  TOR  @ MIN   Pick: MIN   79.4%  Spread: +16.4  Conf: High (0.915)  [INJ: -35H/-70A]
  DET  @ SAS   Pick: DET   67.7%  Spread: -4.0  Conf: Medium (0.737)
  CHI  @ PHX   Pick: CHI   61.5%  Spread: -5.1  Conf: Medium (0.608) [INJ: -70H/-180A]
  LAL  @ DEN   Pick: LAL   50.3%  Spread: +6.4  Conf: Medium (0.56)  [INJ: -159H/-7A]
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/JManson2038/NBA_AI_agent_Game_Predictor.git
cd NBA_AI_agent_Game_Predictor

# 2. Install dependencies
pip install -r requirements.txt
pip install torch

# 3. Copy config and add your settings
cp config.template.py config.py

# 4. Train the main models (first run ~5 min)
python run.py train

# 5. Train the player value neural network
python run.py train-nn

# 6. Sync rosters
python run.py roster

# 7. Run today
python run.py schedule
```

 

---

## Player Tiers

The `PlayerValueNN` scores every player 0-1 using box score stats and win impact metrics (50/50 weighted), then maps them to tiers used for injury Elo penalties:

| Tier | Score | Elo Penalty | Example Players |
|------|-------|-------------|-----------------|
| Franchise | 0.50+ | −90 to −120 | SGA, Luka, Tatum |
| Star | 0.40–0.50 | −60 to −85 | Jaylen Brown, Bam Adebayo |
| Key Rotation | 0.28–0.40 | −30 to −55 | Jrue Holiday, Al Horford |
| Bench | 0.14–0.28 | −10 to −25 | Role players |
| Two-Way | 0.00–0.14 | −5 to −10 | Fringe roster |

**Hard cap:** No team loses more than 180 Elo from injuries regardless of how many players are out.

```bash
python run.py train-nn --team OKC       # one team's rankings
python run.py train-nn --tier Franchise  # all franchise players league-wide
python run.py train-nn --tier Star       # all stars league-wide
```

---

## All Commands

```bash
# Predictions
python run.py schedule              # today's games + predictions
python run.py results               # fill last night's results
python run.py results --date 2026-03-04
python run.py summary               # running ROI
python run.py preview               # schedule only, no predictions
python run.py main --home BOS --away LAL
python run.py main --games OKC:BOS LAL:GSW
python run.py rankings              # Elo power rankings

# Training
python run.py train                 # train LR + XGBoost models
python run.py train-nn              # train player value NN
python run.py train-nn --force      # re-fetch player data

# Maintenance
python run.py roster                # sync all 30 rosters
python run.py roster --team LAL     # sync one team
python run.py roster --check        # check cache staleness
python run.py injuries              # update injuries.json from ESPN

# Analysis
python run.py backtest              # compare Model A / B / C
python run.py backtest --season 2023-24
python run.py backtest --min-edge 0.04
```

---

## Project Structure

```
NBA_AI_agent_Game_Predictor/
├── run.py                       entry point for all commands
├── migrate.py                   one-time migration script
├── config.py                    gitignored — add settings here
├── config.template.py           safe to commit — copy to config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_agent.py        fetches and engineers NBA game data
│   │   ├── team_strength_agent.py  Elo rating system
│   │   ├── matchup_agent.py     pace, shooting, momentum analysis
│   │   ├── prediction_agent.py  LR + XGBoost ensemble
│   │   ├── confidence_agent.py  prediction reliability scoring
│   │   ├── injury_agent.py      calibrated tiered Elo penalties
│   │   └── orchestrator.py      combines all agents
│   ├── models/
│   │   ├── __init__.py
│   │   └── player_value_nn.py   neural network player importance
│   └── utils/
│       ├── __init__.py
│       └── evaluate.py          accuracy, Brier score, calibration
│
├── scripts/
│   ├── main.py                  CLI predictions
│   ├── train.py                 training pipeline
│   ├── train_player_nn.py       player NN training and tier viewer
│   ├── schedule.py              auto-fetch schedule and predict
│   ├── roster_sync.py           detect trades and roster changes
│   ├── injury_updater.py        scrape ESPN injury report
│   ├── prediction_tracker.py    manual prediction logging
│   └── backtest_injury.py       A/B/C injury model comparison
│
├── data/
│   ├── injuries.json            current injury statuses
│   └── predictions_log.json     all logged predictions and results
│
├── cache/                       auto-generated, gitignored
└── models/                      trained model files, gitignored
```

---
| Date | Record | Units | ROI |
|------|--------|-------|-----|
| 2026-03-03 | 8-2 | +5.27 | +52.7% |
| 2026-03-04 | 2-4 | -2.18 | -36.3% |
| 2026-03-05 | 3-3 | -0.27 | -4.5% |
| 2026-03-06 | 5-2 | +2.09 | +29.9% |
| 2026-03-07 | 4-2 | +1.55 | +25.8% |
| 2026-03-08 | 6-4 | +1.45 | +14.5% |
| 2026-03-09 | 3-2 | +0.73 | +14.6% |
| 2026-03-10 | 4-7 | -3.36 | -30.6% |
| 2026-03-15 | 3-3 | -0.27 | -4.5% |
| 2026-03-16 | 5-3 | +1.55 | +19.4% |
| 2026-03-17 | 5-3 | +1.55 | +19.4% |
| 2026-03-18 | 6-3 | +2.45 | +27.2% |
| 2026-03-19 | 3-5 | -2.27 | -28.4% |
| 2026-03-20 | 5-1 | +3.55 | +59.2% |
| **Total** | **62-44** | **+12.84** | **+12.1%** |

*High: 17/28 (60.7%) — +4.45u, +15.9% ROI*

## Tech Stack

Python · pandas · scikit-learn · XGBoost · PyTorch · nba_api
