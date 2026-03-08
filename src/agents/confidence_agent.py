
import numpy as np
import config

class ConfidenceAgent:


    def __init__(self):
       
        self.calibration_ceiling = 0.75
        self.calibration_compression = 0.4

        
        self.disagreement_threshold = 0.20

    def calibrate_probability(self, p):
    
        if p > self.calibration_ceiling:
            p = self.calibration_ceiling + (p - self.calibration_ceiling) * self.calibration_compression
        elif p < (1 - self.calibration_ceiling):
            floor = 1 - self.calibration_ceiling
            p = floor - (floor - p) * self.calibration_compression
        return round(np.clip(p, 0.01, 0.99), 4)

    def check_model_disagreement(self, lr_prob, xgb_prob):
        
        gap = abs(float(lr_prob) - float(xgb_prob))
        should_downgrade = gap > self.disagreement_threshold
        return {
            "model_gap": round(gap, 3),
            "models_conflict": should_downgrade,
        }

    def score(self, prediction, game_features=None):
        
        breakdown = {}

        lr_p = float(prediction["lr_prob"])
        xgb_p = float(prediction["xgb_prob"])
        ens_p = float(prediction["ensemble_prob"])

        # Model Agreement (0-1)
        agreement = 1.0 - abs(lr_p - xgb_p)
        breakdown["model_agreement"] = round(agreement, 3)

        # Prediction Decisiveness (0-1) 
        decisiveness = abs(ens_p - 0.5) * 2
        breakdown["decisiveness"] = round(decisiveness, 3)

        # Elo Confidence (0-1)
        elo_conf = 0.5
        if game_features:
            elo_diff = abs(game_features.get("ELO_DIFF", 0))
            elo_conf = min(elo_diff / 300, 1.0)
        breakdown["elo_strength"] = round(elo_conf, 3)

        #Data Completeness (0-1)
        completeness = 1.0
        if game_features:
            total_keys = len(game_features)
            nan_count = sum(
                1 for v in game_features.values()
                if v is None or (isinstance(v, float) and np.isnan(v))
            )
            completeness = 1.0 - (nan_count / max(total_keys, 1))
        breakdown["data_completeness"] = round(completeness, 3)

        # Volatility Penalty (0-1, higher is calmer)
        volatility_score = 0.7
        if game_features:
            home_mom = abs(game_features.get("HOME_MOMENTUM", 0))
            away_mom = abs(game_features.get("AWAY_MOMENTUM", 0))
            avg_mom = (home_mom + away_mom) / 2
            volatility_score = max(0, 1.0 - (avg_mom / 10))
        breakdown["stability"] = round(volatility_score, 3)

        # Weighted combination 
        weights = {
            "model_agreement": 0.30,   # Increased from 0.25 
            "decisiveness": 0.25,      # Decreased from 0.30
            "elo_strength": 0.20,
            "data_completeness": 0.10,
            "stability": 0.15,
        }

        conf_score = sum(breakdown[k] * weights[k] for k in weights)
        conf_score = round(np.clip(conf_score, 0, 1), 3)

        #  Downgrade if models conflict 
        disagreement = self.check_model_disagreement(lr_p, xgb_p)
        breakdown["model_gap"] = disagreement["model_gap"]
        breakdown["models_conflict"] = disagreement["models_conflict"]

        if disagreement["models_conflict"]:
            conf_score = round(conf_score * 0.75, 3)  # 25% penalty
            breakdown["disagreement_penalty"] = True

        #  Label 
        if conf_score >= config.CONFIDENCE_HIGH:
            label = "High"
        elif conf_score >= config.CONFIDENCE_MEDIUM:
            label = "Medium"
        else:
            label = "Low"

        #  Calibrate probability 
        calibrated_prob = self.calibrate_probability(ens_p)

        return {
            "confidence_score": conf_score,
            "confidence_label": label,
            "raw_probability": round(ens_p, 4),
            "calibrated_probability": calibrated_prob,
            "breakdown": breakdown,
        }