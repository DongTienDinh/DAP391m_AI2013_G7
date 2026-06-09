import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, List

class ModelEvaluator:
    """Handles walk-forward cross-validation and metric calculation."""

    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray, log_target: bool = True) -> Dict[str, float]:
        """Performs walk-forward CV and returns averaged metrics."""
        metrics = {'RMSE': [], 'MAE': []}
        
        for train_idx, val_idx in self.tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            # Inverse log transform if needed
            if log_target:
                p = np.expm1(y_pred)
                t = np.expm1(y_val)
            else:
                p, t = y_pred, y_val
            
            metrics['RMSE'].append(np.sqrt(mean_squared_error(t, p)))
            metrics['MAE'].append(mean_absolute_error(t, p))
            
        return {k: float(np.mean(v)) for k, v in metrics.items()}
