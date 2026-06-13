import warnings
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="No further splits")
warnings.filterwarnings("ignore", message="does not have valid feature names")


class ModelEvaluator:
    """Handles walk-forward cross-validation and full set of forecasting metrics."""

    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def evaluate(
        self, model: Any, x: np.ndarray, y: np.ndarray, log_target: bool = True
    ) -> dict[str, float]:
        """Performs walk-forward CV and returns averaged metrics."""
        metrics: dict[str, list[float]] = {
            "RMSE": [], "MAE": [], "WAPE": [], "sMAPE": [], "MASE": [],
        }

        for train_idx, val_idx in self.tscv.split(x):
            x_tr, x_val = x[train_idx], x[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_val)

            # Inverse log transform
            if log_target:
                p = np.expm1(y_pred)
                t = np.expm1(y_val)
                y_train_actual = np.expm1(y_tr)
            else:
                p, t = y_pred, y_val
                y_train_actual = y_tr

            p = np.maximum(p, 0)  # No negative revenue

            rmse = np.sqrt(mean_squared_error(t, p))
            mae = mean_absolute_error(t, p)

            # WAPE: sum|error| / sum|actual| * 100
            sum_actual = np.sum(np.abs(t))
            wape = (np.sum(np.abs(t - p)) / sum_actual * 100) if sum_actual > 0 else np.nan

            # sMAPE: mean(2|e|/(|a|+|p|)) * 100
            denom = np.abs(t) + np.abs(p)
            smape = np.mean(2.0 * np.abs(t - p) / np.where(denom == 0, 1.0, denom)) * 100

            # MASE: MAE / in-sample naive 1-step MAE
            naive_errors = np.abs(np.diff(y_train_actual))
            naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
            mase = mae / naive_mae if (not np.isnan(naive_mae) and naive_mae > 0) else np.nan

            metrics["RMSE"].append(rmse)
            metrics["MAE"].append(mae)
            metrics["WAPE"].append(wape)
            metrics["sMAPE"].append(smape)
            metrics["MASE"].append(mase)

        return {k: float(np.nanmean(v)) for k, v in metrics.items()}
