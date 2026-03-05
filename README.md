#  NBA AI Agent Game Predictor

A **multi-agent machine learning system** that predicts NBA game outcomes using specialized AI agents for team strength, matchup analysis, and confidence scoring.

## Architecture

```
Data Agent → Team Strength Agent → Matchup Agent
                    ↓
          Prediction Model Agent
                    ↓
            Confidence Agent
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
| **Orchestrator** | Combines all agents into a final prediction with explanation |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data & train models (first run takes ~5 min for API calls)
python train.py

# 3. Predict a game
python main.py --home BOS --away LAL

# 4. Interactive mode
python main.py

# 5. View Elo power rankings
python main.py --rankings

# 6. Backtest a season
python main.py --backtest 2023-24
```

## Sample Output

```
==================================================
 LAL  @  BOS
==================================================
   Predicted Winner:  BOS
   Win Probability:   64.2%
   Spread:            -5.2
   Confidence:        High (0.782)

  Key Factors:
    • Home team Elo advantage (+87)
    • Home team shooting better from 3PT
    • Home team on a hot streak

  Model Detail:
    LR:  0.612
    XGB: 0.655
==================================================
```

## Evaluation Metrics

- **Accuracy** vs home-team-always baseline
- **Log Loss** (probabilistic calibration)
- **Brier Score** (probability accuracy)
- **Calibration curve** (predicted prob vs actual win rate)
- **Confidence tier analysis** (accuracy at High/Medium/Low confidence)

## Tech Stack

Python, pandas, scikit-learn, XGBoost, nba_api

## Future Extensions

- [ ] Streamlit dashboard
- [ ] SHAP explanations
- [ ] Monte Carlo season simulation
- [ ] Live injury integration
- [ ] Betting line comparison
