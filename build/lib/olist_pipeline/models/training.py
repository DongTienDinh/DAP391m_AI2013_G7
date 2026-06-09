import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.olist_pipeline.core.logging_setup import setup_logger
from src.olist_pipeline.models.factory import ModelFactory
from src.olist_pipeline.models.evaluator import ModelEvaluator

logger = setup_logger("model_service")

class ModelingService:
    """Service for training, evaluating, and promoting ML models."""

    def __init__(self, training_config: Dict[str, Any], reports_dir: Path):
        self.config = training_config
        self.reports_dir = reports_dir
        self.evaluator = ModelEvaluator(n_splits=training_config.get('n_splits', 5))

    def run_training_pipeline(self, data_path: Path) -> None:
        """Benchmarks all models and saves results."""
        logger.info("Starting Modeling Service...")
        
        # 1. Load and Prepare
        df = pd.read_csv(data_path)
        X, y, feature_cols = self._prepare_features(df)
        
        # 2. Benchmark
        results = []
        for name, params in self.config.get('models', {}).items():
            logger.info(f"Evaluating {name}...")
            model = ModelFactory.create_pipeline(name, params, self.config.get('random_state', 42))
            metrics = self.evaluator.evaluate(model, X, y, self.config.get('log_target', True))
            results.append({"model": name, **metrics})
            
        # 3. Save Leaderboard
        self._save_leaderboard(results)
        logger.info("Modeling Service completed.")

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Selects features and transforms target."""
        exclude = self.config.get('feature_selection', {}).get('exclude', [])
        feature_cols = [c for c in df.columns if c not in exclude + ['target_next_revenue']]
        
        X = df[feature_cols].values
        y = df['target_next_revenue'].values
        
        if self.config.get('log_target', True):
            y = np.log1p(y)
            
        return X, y, feature_cols

    def _save_leaderboard(self, results: List[Dict[str, Any]]) -> None:
        """Saves evaluation metrics to CSV."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        lb_df = pd.DataFrame(results).sort_values("RMSE")
        lb_path = self.reports_dir / "model_leaderboard.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"Model leaderboard saved to {lb_path}")
