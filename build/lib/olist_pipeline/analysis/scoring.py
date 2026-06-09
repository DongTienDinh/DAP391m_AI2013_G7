import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.olist_pipeline.core.logging_setup import setup_logger
from src.olist_pipeline.analysis.optimizer import EntropyOptimizer
from src.olist_pipeline.analysis.engine import EPSComponentEngine

logger = setup_logger("scoring_service")

class ScoringService:
    """Service for calculating state-level EPS rankings."""

    def __init__(self, inference_config: Dict[str, Any], output_dir: Path):
        self.config = inference_config
        self.output_dir = output_dir

    def run_scoring_pipeline(self, features_path: Path, pred_path: Path) -> None:
        """Main entry point for EPS calculation."""
        logger.info("Starting Scoring Service...")
        
        # 1. Load data
        df_state = self._prepare_state_metrics(features_path, pred_path)
        
        # 2. Calculate Components
        raw_components = self._compute_raw(df_state)
        
        # 3. Optimize Weights
        w_star = self._optimize(raw_components)
        
        # 4. Final Ranking
        results = self._rank(raw_components, w_star)
        
        # 5. Save
        self._save(results, w_star)
        logger.info("Scoring Service completed.")

    def _prepare_state_metrics(self, features_path: Path, pred_path: Path) -> pd.DataFrame:
        """Aggregates features and merges with predictions."""
        # Simplified aggregation logic
        df = pd.read_csv(features_path)
        pred = pd.read_csv(pred_path)
        return df.groupby("customer_state").last().reset_index()

    def _compute_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes raw components using the Engine."""
        out = pd.DataFrame({"customer_state": df["customer_state"]})
        # Logic using EPSComponentEngine...
        return out

    def _optimize(self, df: pd.DataFrame) -> Dict[str, float]:
        """Runs the Entropy Optimizer."""
        # Simplified constraint mapping
        constraints = [(0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.05, 0.2)]
        # w = EntropyOptimizer.find_optimal_weights(matrix, constraints)
        return {"PD": 0.3, "GP": 0.2, "PG": 0.3, "MMI": 0.2}

    def _rank(self, df: pd.DataFrame, w_star: Dict[str, float]) -> pd.DataFrame:
        """Calculates final EPS score and rank."""
        df['EPS_score'] = 75.0 # Dummy
        df['EPS_rank'] = 1 # Dummy
        return df

    def _save(self, df: pd.DataFrame, w_star: Dict[str, float]) -> None:
        """Saves CSV results and weights JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "eps_results.csv", index=False)
        with open(self.output_dir / "w_star.json", "w") as f:
            json.dump({"w_star": w_star, "gamma": self.config.get('gamma', 0.2)}, f, indent=2)
