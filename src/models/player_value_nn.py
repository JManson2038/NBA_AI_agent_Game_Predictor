import json
import time
import numpy as np
import joblib
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[PlayerValueNN] PyTorch not found. Run: pip install torch")

try:
    from nba_api.stats.endpoints import (
        LeagueDashPlayerStats,
        PlayerDashboardByGeneralSplits,
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

CACHE_DIR = Path("cache")
MODEL_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}
def fetch_with_retry(endpoint_fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return endpoint_fn()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt+1} failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e

# ── Tier definitions (score thresholds) ──────────────────────────
TIERS = {
    "Franchise":    (0.49, 1.00),   # Top 1-2 players, MVP-caliber (SGA, Tatum, LeBron)
    "Star":         (0.40, 0.49),   # Clear starter, high usage
    "Key Rotation": (0.29, 0.40),   # Solid starter or key bench piece
    "Bench":        (0.15, 0.29),   # Regular bench contributor
    "Two-Way":      (0.00, 0.15),   # Fringe roster / two-way contract
}
TIER_ELO_RANGE = {
    "Franchise":    (90, 120),
    "Star":         (60,  85),
    "Key Rotation": (30,  55),
    "Bench":        (10,  25),
    "Two-Way":      ( 5,  10),
}

# Input features the NN uses
FEATURE_NAMES = [
    # Box score (per game)
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
    # Efficiency
    "USG_PCT", "TS_PCT", "FG_PCT",
    # Win impact
    "PLUS_MINUS", "WIN_SHARES_PROXY", "BPM_PROXY",
    # Role context
    "TEAM_MIN_SHARE",   # player mins / team total mins
    "TEAM_PTS_SHARE",   # player pts / team total pts
]

N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────
#  Neural Network Architecture
# ─────────────────────────────────────────────────────────────────

class PlayerValueNet(nn.Module):
    """
    3-layer feedforward network.
    Input:  N_FEATURES player stats
    Output: importance score (0-1, sigmoid)

    Architecture chosen to be small enough to train on ~500 players
    without overfitting, while capturing non-linear interactions
    between stats (e.g. high mins + low usage = role player,
    low mins + high usage = injury-limited star).
    """

    def __init__(self, input_dim=N_FEATURES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid(),          # Output in [0, 1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



#  Data Collection

class PlayerDataCollector:
    """Fetches and caches player stats from nba_api."""

    def __init__(self, seasons=None):
        self.seasons = seasons or ["2023-24", "2024-25", "2025-26"]

    def fetch_league_stats(self, season, force=False):
        """Fetch all player per-game stats for a season."""
        cache_file = CACHE_DIR / f"player_stats_{season}.json"

        if cache_file.exists() and not force:
            with open(cache_file) as f:
                return json.load(f)

        if not NBA_API_AVAILABLE:
            print(f"[PlayerDataCollector] nba_api unavailable for {season}")
            return []

        try:
            print(f"[PlayerDataCollector] Fetching {season} player stats...")

            # Per-game base stats
            base = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base",
                headers=HEADERS,
                timeout=120,
            )
            time.sleep(0.8)
            base_df = base.get_data_frames()[0]

            # Advanced stats (usage, true shooting, etc.)
            adv = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                headers=HEADERS,
                timeout=120,
            )
            time.sleep(0.8)
            adv_df = adv.get_data_frames()[0]

            # Merge on player ID
            merged = base_df.merge(
                adv_df[["PLAYER_ID", "USG_PCT", "TS_PCT", "NET_RATING", "PIE"]],
                on="PLAYER_ID", how="left"
            )

            # Filter out players with very few minutes (noise)
            merged = merged[merged["MIN"] >= 8].copy()

            # Compute team-relative context features
            team_totals = merged.groupby("TEAM_ID").agg(
                TEAM_MIN=("MIN", "sum"),
                TEAM_PTS=("PTS", "sum")
            ).reset_index()
            merged = merged.merge(team_totals, on="TEAM_ID", how="left")
            merged["TEAM_MIN_SHARE"] = merged["MIN"] / merged["TEAM_MIN"].clip(lower=1)
            merged["TEAM_PTS_SHARE"] = merged["PTS"] / merged["TEAM_PTS"].clip(lower=1)

            # Win shares proxy: PIE * minutes (PIE = player impact estimate)
            merged["WIN_SHARES_PROXY"] = (
                merged.get("PIE", 0).fillna(0) * merged["MIN"] / 48
            )

            # BPM proxy: net rating adjusted for usage
            merged["BPM_PROXY"] = (
                merged.get("NET_RATING", 0).fillna(0) *
                merged.get("USG_PCT", 0.2).fillna(0.2)
            )

            records = merged.to_dict(orient="records")
            with open(cache_file, "w") as f:
                json.dump(records, f)

            print(f"[PlayerDataCollector] {len(records)} players cached for {season}.")
            return records

        except Exception as e:
            print(f"[PlayerDataCollector] Error fetching {season}: {e}")
            return []

    def collect_all(self, force=False):
        """Collect stats across all seasons into one list."""
        all_records = []
        for season in self.seasons:
            records = self.fetch_league_stats(season, force=force)
            for r in records:
                r["SEASON"] = season
            all_records.extend(records)
        print(f"[PlayerDataCollector] Total records: {len(all_records)}")
        return all_records


# ─────────────────────────────────────────────────────────────────
#  Label Generation (soft targets from domain rules)
# ─────────────────────────────────────────────────────────────────

def generate_importance_label(record):
    """
    Generate a soft importance score (0-1) from domain rules.
    This is the training target — the NN learns to predict this
    from raw stats, capturing non-linear combinations.

    Formula weights both counting stats AND win impact equally,
    as per user spec. Normalized per component to [0,1].
    """
    # ── Box score component (0-1) ──
    pts_norm   = min(record.get("PTS", 0) / 35, 1.0)
    min_norm   = min(record.get("MIN", 0) / 40, 1.0)
    usg_norm   = min(record.get("USG_PCT", 0) / 0.40, 1.0)
    reb_norm   = min(record.get("REB", 0) / 15, 1.0)
    ast_norm   = min(record.get("AST", 0) / 12, 1.0)
    stl_norm   = min(record.get("STL", 0) / 3, 1.0)
    blk_norm   = min(record.get("BLK", 0) / 4, 1.0)
    tov_penalty = min(record.get("TOV", 0) / 5, 1.0) * 0.3  # Penalize turnovers

    box_score = (
        0.30 * pts_norm +
        0.20 * min_norm +
        0.15 * usg_norm +
        0.10 * reb_norm +
        0.10 * ast_norm +
        0.08 * stl_norm +
        0.07 * blk_norm -
        tov_penalty
    )
    box_score = np.clip(box_score, 0, 1)

    # ── Win impact component (0-1) ──
    ws_norm  = np.clip(record.get("WIN_SHARES_PROXY", 0) / 0.3, 0, 1)
    bpm_norm = np.clip((record.get("BPM_PROXY", 0) + 5) / 15, 0, 1)
    ts_norm  = np.clip((record.get("TS_PCT", 0.5) - 0.45) / 0.25, 0, 1)
    pts_shr  = np.clip(record.get("TEAM_PTS_SHARE", 0) / 0.30, 0, 1)
    min_shr  = np.clip(record.get("TEAM_MIN_SHARE", 0) / 0.25, 0, 1)

    win_impact = (
        0.30 * ws_norm +
        0.25 * bpm_norm +
        0.20 * ts_norm +
        0.15 * pts_shr +
        0.10 * min_shr
    )
    win_impact = np.clip(win_impact, 0, 1)

    # ── Combined: 50/50 as per spec ──
    score = 0.50 * box_score + 0.50 * win_impact
    return round(float(np.clip(score, 0.01, 0.99)), 4)


def score_to_tier(score):
    """Map importance score to tier name."""
    for tier, (lo, hi) in TIERS.items():
        if lo <= score < hi:
            return tier
    return "Two-Way"


# ─────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────

def build_feature_vector(record):
    """Convert a player record dict to a normalized feature vector."""
    def safe(key, default=0.0):
        v = record.get(key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    return [
        safe("PTS") / 40,
        safe("REB") / 20,
        safe("AST") / 15,
        safe("STL") / 4,
        safe("BLK") / 5,
        safe("TOV") / 6,
        safe("MIN") / 42,
        safe("USG_PCT") / 0.45,
        safe("TS_PCT") / 0.80,
        safe("FG_PCT") / 0.70,
        np.clip((safe("PLUS_MINUS") + 20) / 40, 0, 1),
        np.clip(safe("WIN_SHARES_PROXY") / 0.4, 0, 1),
        np.clip((safe("BPM_PROXY") + 5) / 15, 0, 1),
        np.clip(safe("TEAM_MIN_SHARE") / 0.30, 0, 1),
        np.clip(safe("TEAM_PTS_SHARE") / 0.35, 0, 1),
    ]


class PlayerValueNN:
    """
    Main class for training and using the player importance NN.
    Usage:
        nn = PlayerValueNN()
        nn.train()                              # train from data
        score = nn.predict_player(record)       # score one player
        tier  = nn.get_tier(record)             # get tier label
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = MODEL_DIR / "player_value_nn.pt"
        self.collector = PlayerDataCollector()

    def train(self, force_fetch=False, epochs=120, lr=0.001):
        """Train the NN on collected player data."""
        if not TORCH_AVAILABLE:
            print("[PlayerValueNN] PyTorch required. pip install torch")
            return

        print("[PlayerValueNN] Collecting player data...")
        records = self.collector.collect_all(force=force_fetch)

        if not records:
            print("[PlayerValueNN] No data to train on.")
            return

        # Build feature matrix and labels
        X, y = [], []
        for r in records:
            try:
                fv = build_feature_vector(r)
                label = generate_importance_label(r)
                X.append(fv)
                y.append(label)
            except Exception:
                continue

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        print(f"[PlayerValueNN] Training on {len(X)} player-season records...")

        # Train/val split
        n = len(X)
        idx = torch.randperm(n)
        split = int(n * 0.85)
        tr_idx, val_idx = idx[:split], idx[split:]

        train_ds = TensorDataset(X[tr_idx], y[tr_idx])
        val_ds   = TensorDataset(X[val_idx], y[val_idx])
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=64)

        self.model = PlayerValueNet(input_dim=N_FEATURES)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            self.model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            if (epoch + 1) % 20 == 0:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_dl:
                        val_losses.append(loss_fn(self.model(xb), yb).item())
                val_loss = np.mean(val_losses)
                print(f"  Epoch {epoch+1:3d}/{epochs}  val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_path)

        # Reload best checkpoint
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.is_trained = True
        print(f"[PlayerValueNN] Training complete. Best val loss: {best_val_loss:.4f}")

    def load(self):
        """Load a previously trained model."""
        if not TORCH_AVAILABLE:
            return False
        if not self.model_path.exists():
            return False
        self.model = PlayerValueNet(input_dim=N_FEATURES)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()
        self.is_trained = True
        return True

    def predict_player(self, record):
        """
        Score a single player record.
        Returns importance score (0-1).
        Falls back to rule-based label if NN not trained.
        """
        if not self.is_trained or not TORCH_AVAILABLE:
            # Graceful fallback to rule-based scoring
            return generate_importance_label(record)

        fv = build_feature_vector(record)
        x = torch.tensor([fv], dtype=torch.float32)
        with torch.no_grad():
            score = self.model(x).item()
        return round(float(score), 4)

    def get_tier(self, record):
        """Get tier label for a player."""
        score = self.predict_player(record)
        return score_to_tier(score), score

    def rank_team(self, player_records):
        """
        Rank all players on a team by importance.
        Returns sorted list of (player_name, tier, score).
        """
        ranked = []
        for r in player_records:
            score = self.predict_player(r)
            tier = score_to_tier(score)
            ranked.append({
                "player": r.get("PLAYER_NAME", r.get("PLAYER", "Unknown")),
                "tier": tier,
                "score": score,
                "mpg": round(r.get("MIN", 0), 1),
                "ppg": round(r.get("PTS", 0), 1),
                "usg": round(r.get("USG_PCT", 0) * 100, 1),
            })
        return sorted(ranked, key=lambda x: -x["score"])

    def print_team_rankings(self, team_abbr, player_records):
        """Pretty print team player rankings."""
        ranked = self.rank_team(player_records)
        print(f"\n  PLAYER VALUE RANKINGS — {team_abbr}")
        print("  " + "─" * 58)
        print(f"  {'Player':<24} {'Tier':<14} {'Score':>6}  {'MPG':>5}  {'PPG':>5}  {'USG%':>5}")
        print("  " + "─" * 58)

        for p in ranked:
            bar = "█" * int(p["score"] * 20)
            print(
                f"  {p['player']:<24} {p['tier']:<14} "
                f"{p['score']:>6.3f}  {p['mpg']:>5.1f}  "
                f"{p['ppg']:>5.1f}  {p['usg']:>5.1f}"
            )
        print("  " + "─" * 58)