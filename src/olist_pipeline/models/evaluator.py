from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class ModelEvaluator:
    """Handles walk-forward cross-validation and metric calculation."""

    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def evaluate(
        self, model: Any, x: np.ndarray, y: np.ndarray, log_target: bool = True
    ) -> dict[str, float]:
        """Performs walk-forward CV and returns averaged metrics."""
        metrics: dict[str, list[float]] = {"RMSE": [], "MAE": []}

        for train_idx, val_idx in self.tscv.split(x):
            x_tr, x_val = x[train_idx], x[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_val)
            # Inverse log transform if needed
            if log_target:
                p = np.expm1(y_pred)
                t = np.expm1(y_val)
            else:
                p, t = y_pred, y_val

            metrics["RMSE"].append(np.sqrt(mean_squared_error(t, p)))
            metrics["MAE"].append(mean_absolute_error(t, p))

        return {k: float(np.mean(v)) for k, v in metrics.items()}
