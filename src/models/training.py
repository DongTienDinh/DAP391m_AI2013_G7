from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger, short_path
from src.models.evaluator import ModelEvaluator
from src.models.factory import ModelFactory

logger = setup_logger("model_service")


class ModelingService:
    """Service for training, evaluating, and promoting ML models."""

    def __init__(self, training_config: dict[str, Any], reports_dir: Path, processed_dir: Path):
        self.config = training_config
        self.reports_dir = reports_dir
        self.processed_dir = processed_dir
        self.evaluator = ModelEvaluator(n_splits=training_config.get("n_splits", 5))

    def run_training_pipeline(self, data_path: Path) -> None:
        logger.info("Starting Modeling Service...")

        df = pd.read_csv(data_path)
        x, y, y_model, feature_cols = self._prepare_features(df)
        models = ModelFactory.create_all(self.config.get("models", {}), self.config.get("random_state", 42))

        # Benchmark
        results = []
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            metrics = self.evaluator.evaluate(model, x, y_model, self.config.get("log_target", True))
            results.append({"model": name, **metrics})

        # Add skill score
        if results:
            baseline_rmse = next(r["RMSE"] for r in results if r["model"] == "Linear Regression (Baseline)")
            for r in results:
                r["SS_RMSE"] = 1.0 - (r["RMSE"] / baseline_rmse) if baseline_rmse > 0 else np.nan

        # Save leaderboard
        self._save_leaderboard(results)

        # Identify best model and save predictions
        best_model_name = min(results, key=lambda r: r["RMSE"])["model"]
        logger.info(f"Champion model: {best_model_name}")
        self._generate_predictions(best_model_name, models, df, x, y, y_model, feature_cols)

        logger.info("Modeling Service completed.")

    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        # Add legacy features matching the original notebook
        df["state_code"] = df["customer_state"].astype("category").cat.codes
        grp = df.groupby("customer_state")
        df["revenue_std_4"] = grp["revenue"].transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).std()
        ).fillna(0)
        df["revenue_momentum"] = (df["revenue_lag_1"] - df["revenue_lag_4"]).fillna(0)

        exclude = self.config.get("feature_selection", {}).get("exclude", [])
        feature_cols = [c for c in df.columns if c not in exclude + ["target_next_revenue"]]

        x = df[feature_cols].values
        y = df["target_next_revenue"].values
        y_model = np.log1p(y) if self.config.get("log_target", True) else y

        return x, y, y_model, feature_cols

    def _save_leaderboard(self, results: list[dict[str, Any]]) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        lb_df = pd.DataFrame(results).sort_values("RMSE")
        lb_path = self.reports_dir / "model_leaderboard.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"Model leaderboard saved to {short_path(lb_path)}")

    def _generate_predictions(
        self,
        best_model_name: str,
        models: dict[str, Any],
        df_train: pd.DataFrame,
        x_train: np.ndarray,
        y_train: np.ndarray,
        y_train_model: np.ndarray,
        feature_cols: list[str],
    ) -> None:
        """Generates next-week revenue predictions using the champion model."""
        pred_path = self.processed_dir / "prediction_data.csv"
        if not pred_path.exists():
            logger.warning(f"Prediction data not found at {pred_path}, skipping future predictions.")
            return

        pred_out_path = self.processed_dir / "predicted_next_week_revenue.csv"
        logger.info(f"Generating predictions with champion: {best_model_name}")

        df_pred = pd.read_csv(pred_path)
        df_combined = pd.concat([df_train, df_pred], ignore_index=True)
        df_combined = df_combined.sort_values(["customer_state", "year_week"]).reset_index(drop=True)
        df_combined["state_code"] = df_combined["customer_state"].astype("category").cat.codes

        # Recalculate dynamic features on merged set
        grp = df_combined.groupby("customer_state")
        df_combined["revenue_std_4"] = grp["revenue"].transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).std()
        ).fillna(0)
        df_combined["revenue_momentum"] = (df_combined["revenue_lag_1"] - df_combined["revenue_lag_4"]).fillna(0)

        # Extract rows to predict (NaN target)
        df_pred_final = df_combined[df_combined["target_next_revenue"].isna()].copy()

        # Retrain best model on full training set
        best_model = models[best_model_name]
        best_model.fit(x_train, y_train_model)

        X_pred = df_pred_final[feature_cols].values
        pred_log = best_model.predict(X_pred)
        pred_revenue = np.maximum(np.expm1(pred_log), 0)

        df_result = pd.DataFrame({
            "customer_state": df_pred_final["customer_state"],
            "year_week_current": df_pred_final["year_week"],
            "predicted_next_week_revenue": pred_revenue,
        })
        df_result = df_result.sort_values("predicted_next_week_revenue", ascending=False).reset_index(drop=True)

        df_result.to_csv(pred_out_path, index=False)
        logger.info(f"Saved predictions to {short_path(pred_out_path)}")

        # Log top 5
        top5 = df_result.head(5)
        for _, row in top5.iterrows():
            logger.info(f"  {row['customer_state']}: {row['predicted_next_week_revenue']:.2f}")
