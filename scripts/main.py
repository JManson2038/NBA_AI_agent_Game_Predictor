import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sys
import pandas as pd
from pathlib import Path
from src.agents.data_agent import DataAgent
from src.agents.team_strength_agent import TeamStrengthAgent
from src.agents.matchup_agent import MatchupAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.confidence_agent import ConfidenceAgent
from src.agents.orchestrator import Orchestrator, format_prediction
import config


def load_pipeline():
    # Load all agents and the cached dataset
    data_agent = DataAgent()

    cache_path = Path("cache/matchup_dataset.parquet")
    if not cache_path.exists():
        print("No cached data found. Run python train.py first.")
        sys.exit(1)

    dataset = pd.read_parquet(cache_path)

    strength_agent = TeamStrengthAgent()
    dataset = strength_agent.run(dataset)

    matchup_agent = MatchupAgent()
    dataset = matchup_agent.run(dataset)

    rolling_cols = data_agent.get_feature_columns(dataset)
    elo_cols = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF", "ELO_WIN_PROB"]
    matchup_cols = matchup_agent.get_matchup_feature_columns(dataset)
    feature_cols = list(dict.fromkeys(rolling_cols + elo_cols + matchup_cols))

    prediction_agent = PredictionAgent()
    prediction_agent._load_models()

    orch = Orchestrator(strength_agent=strength_agent)
    orch.prediction_agent = prediction_agent

    return orch, dataset, feature_cols, strength_agent


def predict_matchup(home, away, orch, dataset, feature_cols):
    # Predict a specific matchup
    pred = orch.predict_matchup(home, away, dataset, feature_cols)
    print(format_prediction(pred))


def show_ratings():
    #Show current Elo power rankings
    orch, dataset, feature_cols, strength = load_pipeline()
    ratings = strength.get_current_ratings()

    print("")
    print("=" * 40)
    print("  NBA Elo Power Rankings")
    print("=" * 40)
    for i, (team, elo) in enumerate(ratings.items(), 1):
        bar_len = int((elo - 1350) / 10)
        bar = "#" * max(bar_len, 0)
        print("  " + str(i).rjust(2) + ". " + team + "  " + str(round(elo, 1)).rjust(7) + "  " + bar)
    print("=" * 40)
    print("")


def backtest_season(season):
    """Run predictions on an entire season and evaluate."""
    orch, dataset, feature_cols, strength = load_pipeline()
    from evaluate import evaluate_predictions

    test_df = dataset[dataset["SEASON"] == season].dropna(subset=feature_cols)
    if test_df.empty:
        print("No data for season " + season)
        return

    print("")
    print("Backtesting " + str(len(test_df)) + " games from " + season + "...")
    results = orch.predict_batch(test_df, feature_cols)
    evaluate_predictions(results)


def main():
    parser = argparse.ArgumentParser(
        description="NBA AI Agent Game Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --home BOS --away LAL
  python main.py --games BOS:LAL CHA:DET MIA:UTA
  python main.py --rankings
  python main.py --backtest 2024-25
        """,
    )
    parser.add_argument("--home", type=str, help="Home team abbreviation (e.g. BOS)")
    parser.add_argument("--away", type=str, help="Away team abbreviation (e.g. LAL)")
    parser.add_argument(
        "--games", nargs="+", type=str,
        help="Multiple games as HOME:AWAY pairs (e.g. BOS:LAL CHA:DET MIA:UTA)"
    )
    parser.add_argument("--rankings", action="store_true", help="Show Elo power rankings")
    parser.add_argument("--backtest", type=str, help="Backtest a season (e.g. 2024-25)")

    args = parser.parse_args()

    if args.rankings:
        show_ratings()
    elif args.backtest:
        backtest_season(args.backtest)
    elif args.games:
        # Batch mode: predict multiple games
        orch, dataset, feature_cols, strength = load_pipeline()
        for game in args.games:
            parts = game.upper().replace(" ", "").split(":")
            if len(parts) != 2:
                print("Invalid format: " + game + "  (use HOME:AWAY, e.g. BOS:LAL)")
                continue
            home, away = parts
            predict_matchup(home, away, orch, dataset, feature_cols)
    elif args.home and args.away:
        # Single game mode
        orch, dataset, feature_cols, strength = load_pipeline()
        predict_matchup(args.home.upper(), args.away.upper(), orch, dataset, feature_cols)
    else:
        # Interactive mode
        print("")
        print("NBA AI Agent Game Predictor")
        print("=" * 40)
        print("Enter matchups as: HOME AWAY (e.g. BOS LAL)")
        print("Type 'rankings' for Elo rankings, 'quit' to exit.")
        print("")

        orch, dataset, feature_cols, strength = load_pipeline()

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "rankings":
                ratings = strength.get_current_ratings()
                for i, (team, elo) in enumerate(ratings.items(), 1):
                    print("  " + str(i).rjust(2) + ". " + team + ": " + str(round(elo, 1)))
                continue

            parts = user_input.upper().split()
            if len(parts) != 2:
                print("  Usage: HOME AWAY  (e.g. BOS LAL)")
                continue

            home, away = parts
            predict_matchup(home, away, orch, dataset, feature_cols)


if __name__ == "__main__":
    main()