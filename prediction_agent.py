

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import config

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class PredictionAgent:

    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE
        )
        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.spread_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
        )
        self.feature_cols = []
        self.is_trained = False

    def train(self, df, feature_cols):
        """Train all models on the provided dataset."""
        print("[PredictionAgent] Training models...")
        self.feature_cols = feature_cols

        train_df = df[df["SEASON"] != config.TEST_SEASON].copy()
        train_df = train_df.dropna(subset=feature_cols + ["HOME_WIN"])

        X = train_df[feature_cols].values
        y = train_df["HOME_WIN"].values

        if "HOME_PTS" in train_df.columns and "AWAY_PTS" in train_df.columns:
            y_spread = (train_df["HOME_PTS"] - train_df["AWAY_PTS"]).values
        else:
            y_spread = None

        X_scaled = self.scaler.fit_transform(X)

        self.lr_model.fit(X_scaled, y)
        lr_acc = self.lr_model.score(X_scaled, y)
        print("  Logistic Regression train acc: " + str(round(lr_acc, 4)))

        self.xgb_model.fit(X, y)
        xgb_acc = self.xgb_model.score(X, y)
        print("  XGBoost train acc: " + str(round(xgb_acc, 4)))

        if y_spread is not None:
            self.spread_model.fit(X, y_spread)
            print("  Spread model trained.")

        self.is_trained = True
        self._save_models()
        print("[PredictionAgent] Training complete.")

    def predict(self, X):
        """
        Predict for a feature matrix.
        Returns dict with probabilities from both models + spread.
        """
        if not self.is_trained:
            self._load_models()

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        lr_prob = self.lr_model.predict_proba(X_scaled)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        spread = self.spread_model.predict(X)

        return {
            "lr_prob": lr_prob,
            "xgb_prob": xgb_prob,
            "ensemble_prob": 0.3 * lr_prob + 0.7 * xgb_prob,
            "spread": spread,
        }

    def predict_single(self, features):
        """Predict a single game from a feature dictionary."""
        X = np.array([features.get(c, 0) for c in self.feature_cols])
        X = X.reshape(1, -1)
        return self.predict(X)

    def get_feature_importance(self, top_n=15):
        """Return XGBoost feature importance."""
        if not self.is_trained:
            self._load_models()
        imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.xgb_model.feature_importances_,
        })
        imp = imp.sort_values("importance", ascending=False)
        return imp.head(top_n)

    def _save_models(self):
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(self.lr_model, MODEL_DIR / "lr_model.pkl")
        joblib.dump(self.xgb_model, MODEL_DIR / "xgb_model.pkl")
        joblib.dump(self.spread_model, MODEL_DIR / "spread_model.pkl")
        joblib.dump(self.feature_cols, MODEL_DIR / "feature_cols.pkl")
        print("[PredictionAgent] Models saved.")

    def _load_models(self):
        try:
            self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            self.lr_model = joblib.load(MODEL_DIR / "lr_model.pkl")
            self.xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
            self.spread_model = joblib.load(MODEL_DIR / "spread_model.pkl")
            self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
            self.is_trained = True
            print("[PredictionAgent] Models loaded.")
        except FileNotFoundError:
            raise RuntimeError("Models not found. Run train.py first.")