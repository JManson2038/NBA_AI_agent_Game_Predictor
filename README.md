# NBA AI Agent Game Predictor

A multi-agent machine learning system that predicts NBA game outcomes using Elo ratings, rolling team stats, matchup analysis, injury impact scoring, and a neural network for player importance weighting.

## Architecture

```
Data Agent → Team Strength Agent → Matchup Agent
                    ↓
          Prediction Model Agent (LR + XGBoost)
                    ↓
            Confidence Agent
                    ↓
          Injury Agent (PlayerValueNN)
                    ↓
              Orchestrator → Final Prediction
```

| Agent | Role |
|-------|------|
| **Data Agent** | Fetches game logs from `nba_api`, engineers rolling features |
| **Team Strength Agent** | Maintains Elo ratings with margin-of-victory adjustments |
| **Matchup Agent** | Detects pace, shooting, rebounding, and momentum mismatches |
| **Prediction Agent** | Logistic Regression (baseline) + XGBoost (primary) ensemble |
| **Confidence Agent** | Scores reliability via model agreement, Elo strength, volatility |
| **Injury Agent** | Calibrated tiered Elo penalties based on PlayerValueNN scores |
| **Orchestrator** | Combines all agents into a final prediction with explanation |

## Features

- **Auto schedule fetching** — pulls today's NBA games automatically
- **Auto injury updates** — scrapes ESPN injury report before every prediction
- **Auto result filling** — fetches final scores from NBA scoreboard each night
- **Player value neural network** — rates every player Franchise / Star / Key Rotation / Bench / Two-Way
- **Calibrated injury penalties** — tiered Elo adjustments based on player importance, not flat penalties
- **Closing line value tracking** — measures model edge vs betting market (requires Odds API key)
- **Roster sync** — detects trades and roster moves, auto-fixes injuries.json
- **Prediction tracker** — logs every prediction with confidence tier and running ROI
- **A/B/C model backtest** — compares no-injury vs fixed penalty vs calibrated injury models

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install torch

# 2. Copy config template and add your settings
cp config.template.py config.py

# 3. Fetch data and train models (first run ~5 min)
python train.py

# 4. Train player value neural network
python train_player_nn.py

# 5. Sync rosters
python roster_sync.py

# 6. Run today's predictions
python schedule.py
```

## Daily Workflow

```bash
# Morning — fetch schedule, update injuries, predict all games
python schedule.py

# Evening — auto-fill results from NBA scoreboard
python schedule.py --results

# Check running ROI
python schedule.py --summary
```

## Sample Output

```
  DAL  @ CHA   Pick: DAL   65.7%  Spread: -2.6  Conf: Medium (0.68)  [INJ: -70H/-171A]
  WAS  @ ORL   Pick: ORL   80.0%  Spread: +13.6  Conf: High (0.922)  [INJ: -71H/-0A]
  OKC  @ CHI   Pick: OKC   58.1%  Spread: +0.4   Conf: Medium (0.658) [INJ: -122H/-106A]
```

## Player Tiers

The PlayerValueNN scores every player on a 0-1 scale using box score stats and win impact metrics, then classifies them into tiers used for injury Elo penalties:

| Tier | Score | Elo Penalty | Example |
|------|-------|-------------|---------|
| Franchise | 0.48+ | 90-120 | SGA, Luka, Tatum |
| Star | 0.40-0.48 | 60-85 | Jaylen Brown, Bam |
| Key Rotation | 0.28-0.40 | 30-55 | Jrue Holiday, Al Horford |
| Bench | 0.14-0.28 | 10-25 | Role players |
| Two-Way | 0.00-0.14 | 5-10 | Fringe roster |

```bash
# View player tiers
python train_player_nn.py --team OKC
python train_player_nn.py --tier Franchise
python train_player_nn.py --tier Star
```

## Roster Management

```bash
# Sync all 30 rosters, detect trades, auto-fix injuries.json
python roster_sync.py

# Check one team
python roster_sync.py --team LAL

# Check if cache is stale
python roster_sync.py --check
```

## Backtesting

```bash
# Compare all three injury models
python backtest_injury.py

# High edge games only
python backtest_injury.py --min-edge 0.04

# Specific season
python backtest_injury.py --season 2023-24
```

## Closing Line Value (CLV)

Measures whether the model finds edge vs the betting market.

```bash
# Requires free API key from https://the-odds-api.com
# Add to config.py: ODDS_API_KEY = "your_key_here"

python clv_tracker.py --fetch      # fetch opening lines
python clv_tracker.py --close      # fetch closing lines
python clv_tracker.py --summary    # show CLV analysis
```

## Manual Prediction

```bash
# Single game
python main.py --home BOS --away LAL

# Multiple games
python main.py --games OKC:BOS LAL:GSW MIA:NYK

# Interactive mode
python main.py

# Elo power rankings
python main.py --rankings

# Backtest a season
python main.py --backtest 2024-25
```

## Project Structure

```
nba_game_predictor/
├── main.py                  # CLI entry point
├── schedule.py              # Auto-fetch schedule and predictions
├── train.py                 # Training pipeline
├── train_player_nn.py       # Player value NN training
├── orchestrator.py          # Combines all agents
├── data_agent.py            # NBA data fetching and feature engineering
├── team_strength_agent.py   # Elo rating system
├── matchup_agent.py         # Matchup feature detection
├── prediction_agent.py      # LR + XGBoost ensemble
├── confidence_agent.py      # Prediction reliability scoring
├── injury_agent.py          # Calibrated injury Elo penalties
├── player_value_nn.py       # Neural network player importance
├── injury_updater.py        # Auto-scrape ESPN injuries
├── roster_sync.py           # Roster change detection
├── clv_tracker.py           # Closing line value tracking
├── prediction_tracker.py    # Manual prediction logging
├── backtest_injury.py       # A/B/C model comparison
├── evaluate.py              # Evaluation metrics
├── config.template.py       # Config template (copy to config.py)
├── injuries.json            # Current injury statuses
├── requirements.txt         # Dependencies
├── cache/                   # Auto-generated data cache
└── models/                  # Trained model files
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, PyTorch, nba_api

## Requirements

```
nba_api>=1.4
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
tqdm>=4.65
torch>=2.0
```

## Configuration

Copy `config.template.py` to `config.py` and set:

```python
ODDS_API_KEY = "your_key_here"   # from https://the-odds-api.com
```

All other settings have working defaults.

## Results (Live Tracking)

| Date | Record | Units | ROI |
|------|--------|-------|-----|
| 2026-03-03 | 8-2 | +5.27 | +52.7% |
| 2026-03-04 | 2-4 | -2.18 | -36.3% |
| **Total** | **10-6** | **+3.09** | **+19.4%** |

*High confidence picks: 6-1*