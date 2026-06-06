import numpy as np
import pandas as pd
from typing import Optional

def softclip_positive(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Smooth approximation of max(x, 0): ln(1 + exp(k * x)) / k.
    Clips input value inside exp to avoid overflow.
    """
    return np.log1p(np.exp(np.clip(k * x, -50, 50))) / k

def normalize_zscore_to_01(series: pd.Series) -> pd.Series:
    """
    Two-step normalization:
      1. Z-score: reduces outlier influence.
      2. Min-max to [0,1]: unified scale across components.
    """
    valid = series.dropna()
    if valid.empty:
        return series.fillna(0.0)
    mu, sigma = valid.mean(), valid.std()
    z = (series - mu) / (sigma + 1e-9)
    z_min, z_max = z.min(), z.max()
    return (z - z_min) / (z_max - z_min + 1e-9)
