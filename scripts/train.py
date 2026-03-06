import argparse
import pandas as pd
from src.agents.data_agent import DataAgent
from src.agents.team_strength_agent import TeamStrengthAgent
from src.agents.matchup_agent import MatchupAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.orchestrator import Orchestrator
from src.utils.evaluate import evaluate_predictions
import config

def train(force_fetch=False):
    print("=" * 60)
    print("  NBA AI Agent Game Predictor — Training Pipeline")
    print("=" * 60)

    #  Data Agent 
    data_agent = DataAgent()
    dataset = data_agent.run(force=force_fetch)
    print(f"\n Dataset shape: {dataset.shape}")

    # Team Strength Agent (Elo) 
    strength_agent = TeamStrengthAgent()
    dataset = strength_agent.run(dataset)

    # Print top teams by Elo
    ratings = strength_agent.get_current_ratings()
    print("\n Top 10 Teams by Elo:")
    for i, (team, elo) in enumerate(list(ratings.items())[:10], 1):
        print(f"  {i:2d}. {team}: {elo}")

    # Matchup Agent
    matchup_agent = MatchupAgent()
    dataset = matchup_agent.run(dataset)

    # Assemble feature columns
    rolling_cols = data_agent.get_feature_columns(dataset)
    elo_cols = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF", "ELO_WIN_PROB"]
    matchup_cols = matchup_agent.get_matchup_feature_columns(dataset)
    feature_cols = rolling_cols + elo_cols + matchup_cols

    # Remove any duplicates
    feature_cols = list(dict.fromkeys(feature_cols))

    print(f"\n Total features: {len(feature_cols)}")

    # Clean data 
    # Drop rows with NaN in features (early-season games with no rolling data)
    clean = dataset.dropna(subset=feature_cols + ["HOME_WIN"])
    print(f" Clean rows: {len(clean)} / {len(dataset)}")

    #  Train Prediction Agent
    prediction_agent = PredictionAgent()
    prediction_agent.train(clean, feature_cols)

    # ── Step 7: Show feature importance ──
    importance = prediction_agent.get_feature_importance(top_n=15)
    print("\n Top 15 Feature Importances:")
    for _, row in importance.iterrows():
        bar = "" * int(row["importance"] * 100)
        print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")

    # ── Step 8: Quick test-set evaluation ──
    test_df = clean[clean["SEASON"] == config.TEST_SEASON]
    if len(test_df) > 0:
        from evaluate import evaluate_predictions
        from orchestrator import Orchestrator

        orch = Orchestrator(strength_agent=strength_agent)
        orch.prediction_agent = prediction_agent
        results = orch.predict_batch(test_df, feature_cols)
        evaluate_predictions(results)
    else:
        print("\n No test season data found for evaluation.")

    print("\n Training complete! Run `python main.py` to predict games.")
    return dataset, feature_cols, strength_agent, prediction_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA AI Predictor")
    parser.add_argument("--force", action="store_true", help="Force re-fetch data from API")
    args = parser.parse_args()
    train(force_fetch=args.force)