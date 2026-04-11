# NBA AI Agent Game Predictor

A multi-agent machine learning system that predicts NBA game outcomes daily. Built with Elo ratings, rolling team statistics, matchup analysis, a neural network for player importance scoring, and automated injury tracking.

---

## How It Works

```
NBA API / ESPN / CDN
  └── Data Agent          fetches game logs, engineers rolling features
        └── Team Strength Agent    builds Elo ratings (margin-of-victory adjusted)
              └── Matchup Agent         detects pace, shooting, rebounding mismatches
                    └── Prediction Agent      LR + XGBoost ensemble
                          └── Confidence Agent      scores prediction reliability
                                └── Injury Agent          tier-weighted Elo penalties via PlayerValueNN
                                      └── Orchestrator          final prediction with explanation
```

---

> Predictions are logged daily with full injury context, confidence tier, and spread. Results are auto-filled each evening from the NBA scoreboard via a 3-layer API fallback (Live CDN, ESPN, stats.nba.com).

---

## Daily Workflow

```bash
# Morning: fetch schedule, update injuries, predict all games
python run.py schedule

# Evening: auto-fill results from NBA scoreboard
python run.py results

# Check running ROI
python run.py summary
```

---

## Sample Output

```
  ══════════════════════════════════════════════════════════
  NBA SCHEDULE — Thursday, April 3, 2026
  ══════════════════════════════════════════════════════════

  IND  @ CHA   Pick: CHA   80.0%  Spread: +14.5  Conf: High (0.907)  [INJ: -213H/-14A]
  CHI  @ NYK   Pick: NYK   78.7%  Spread: +13.5  Conf: High (0.850)  [INJ: -241H/-42A]
  BOS  @ MIL   Pick: BOS   79.3%  Spread: -17.5  Conf: High (0.858)  [INJ: -35H/-250A]
  UTA  @ HOU   Pick: HOU   78.5%  Spread: +12.7  Conf: High (0.893)  [INJ: -250H/-45A]
  SAS  @ GSW   Pick: SAS   79.8%  Spread: -9.8   Conf: High (0.917)  [INJ: -35H/-250A]
```

---

## Live Prediction Record

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
| 2026-03-21 | 8-2 | +5.27 | +52.7% |
| 2026-03-22 | 3-2 | +0.73 | +14.6% |
| 2026-03-23 | 6-4 | +1.45 | +14.5% |
| 2026-03-24 | 4-0 | +3.64 | +90.9% |
| 2026-03-25 | 9-3 | +5.18 | +43.2% |
| 2026-03-26 | 3-0 | +2.73 | +90.9% |
| 2026-03-27 | 7-3 | +3.36 | +33.6% |
| 2026-03-28 | 4-2 | +1.64 | +27.3% |
| 2026-03-29 | 5-4 | +0.55 | +6.1% |
| 2026-03-30 | 8-0 | +7.27 | +90.9% |
| 2026-03-31 | 5-2 | +2.55 | +36.4% |
| 2026-04-01 | 7-2 | +4.36 | +48.4% |
| 2026-04-02 | 4-2 | +1.64 | +27.3% |
| 2026-04-03 | 8-1 | +6.27 | +69.7% |
| 2026-04-04 | 2-1 | +0.82 | +27.3% |
| 2026-04-05 | 8-3 | +4.27 | +38.8% |
| 2026-04-06 | 4-1 | +2.64 | +52.7% |
| 2026-04-07 | 9-1 | +7.18 | +71.8% |
| 2026-04-08 | 6-1 | +4.45 | +63.6% |
| 2026-04-09 | 4-2 | +1.64 | +27.3% |
| 2026-04-10 | 11-4 | +6.00 | +40.0% |
| **Total** | **187-84** | **+86.48** | **+31.9%** |

*High: 82/101 (81.2%) | +54.29u | +53.8% ROI*
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

| Tier | Score | Weight | Example Players |
|------|-------|--------|-----------------|
| Franchise | 0.49+ | 1.5x | SGA, Jokic, Giannis, Luka |
| Star | 0.40-0.49 | 1.3x | Tatum, Kawhi, Wembanyama, Curry |
| Key Rotation | 0.29-0.40 | 1.1x | Jrue Holiday, Mikal Bridges |
| Bench | 0.15-0.29 | 0.7x | Role players |
| Two-Way | 0.00-0.15 | 0.5x | Fringe roster |

Penalties are tier-weighted: losing a Franchise player hurts 1.5x more than losing a Key Rotation player. Safety net cap at 250 Elo maximum per team. XGBoost also learns from `BEST_OUT_SCORE_DIFF`, which tracks the quality gap between each team's best missing player (4th in feature importance).

```bash
python run.py train-nn --team OKC       # one team's rankings
python run.py train-nn --tier Franchise  # all franchise players league-wide
python run.py train-nn --tier Star       # all stars league-wide
```

---

## Key Features

- **Multi-agent architecture**: 7 specialized agents (Data, Elo, Matchup, Prediction, Confidence, Injury, Orchestrator)
- **Ensemble model**: Logistic Regression (30%) + XGBoost (70%) with calibrated probabilities
- **PlayerValueNN**: PyTorch neural network scoring player importance from dual-season stats
- **Tier-weighted injuries**: Franchise players penalized 1.5x, bench players 0.7x (replaces flat cap)
- **Historical injury features**: `BEST_OUT_SCORE_DIFF` built from 280K player game logs across 11 seasons (4th in XGBoost feature importance)
- **3-layer API fallback**: Live CDN (fastest) -> ESPN API (any date) -> stats.nba.com (last resort)
- **Stale data detection**: CDN auto-skips when showing yesterday's games
- **Live tracked record**: 217+ games with daily auto-filled results and ROI tracking

---

## Model Performance

```
Accuracy:       66.8%  (baseline: 54.5%)
High Confidence: 78.5%  on 79 picks
XGBoost acc:     74.4%
Calibration:     near-perfect across all probability buckets

Top 5 Feature Importances:
  1. ELO_DIFF              0.1401
  2. ELO_WIN_PROB           0.0651
  3. HOME_ELO               0.0262
  4. BEST_OUT_SCORE_DIFF    0.0221
  5. AWAY_PLUS_MINUS_ROLL   0.0187
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
```

---

## Project Structure

```
NBA_AI_agent_Game_Predictor/
├── run.py                       entry point for all commands
├── config.py                    gitignored settings
├── config.template.py           safe to commit
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── agents/
│   │   ├── data_agent.py        fetches and engineers NBA game data
│   │   ├── team_strength_agent.py  Elo rating system
│   │   ├── matchup_agent.py     pace, shooting, momentum analysis
│   │   ├── prediction_agent.py  LR + XGBoost ensemble
│   │   ├── confidence_agent.py  prediction reliability scoring
│   │   ├── injury_agent.py      tier-weighted Elo penalties
│   │   └── orchestrator.py      combines all agents
│   ├── models/
│   │   └── player_value_nn.py   neural network player importance
│   └── utils/
│       └── evaluate.py          accuracy, Brier score, calibration
│
├── scripts/
│   ├── schedule.py              auto-fetch schedule and predict (CDN/ESPN/stats.nba.com)
│   ├── train.py                 training pipeline
│   ├── train_player_nn.py       player NN training and tier viewer
│   ├── prediction_tracker.py    ROI tracking and summary
│   ├── build_injury_features.py historical injury feature builder
│   ├── roster_sync.py           detect trades and roster changes
│   ├── injury_updater.py        scrape ESPN injury report
│   └── main.py                  CLI predictions and rankings
│
├── data/
│   ├── injuries.json            current injury statuses
│   └── predictions_log.json     all logged predictions and results
│
├── cache/                       auto-generated, gitignored
└── models/                      trained model files, gitignored
```

---

## Tech Stack

Python, pandas, scikit-learn, XGBoost, PyTorch, nba_api, ESPN API