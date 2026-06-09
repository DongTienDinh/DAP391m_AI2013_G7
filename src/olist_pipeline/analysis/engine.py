import numpy as np
import pandas as pd


class EPSComponentEngine:
    """Core logic for calculating expansion score components."""

    @staticmethod
    def calculate_pd(predicted: pd.Series, actual_4w: pd.Series) -> pd.Series:
        """Predicted Demand: Blends forecast and recent actuals."""
        return 0.5 * np.log1p(predicted.fillna(0)) + 0.5 * np.log1p(actual_4w.fillna(0))

    @staticmethod
    def calculate_gp(actual_4w: pd.Series, actual_8w: pd.Series) -> pd.Series:
        """Growth Potential: Cycle-over-cycle change."""
        epsilon = actual_8w.median() * 0.01 if len(actual_8w) > 0 else 1.0
        safe_8w = actual_8w.clip(lower=epsilon)
        return ((actual_4w - safe_8w) / safe_8w).clip(-1, 1).fillna(0)

    @staticmethod
    def calculate_pg(population: pd.Series, gdp_pc: pd.Series, actual_4w: pd.Series) -> pd.Series:
        """Penetration Gap: Theoretical vs. Actual revenue."""
        avg_gdp_pc = np.average(gdp_pc, weights=population)
        gdp_weight = gdp_pc / avg_gdp_pc
        avg_rev_pc = actual_4w.sum() / population.sum()
        expected = population * avg_rev_pc * gdp_weight
        return (expected - actual_4w) / (expected + 1e-9)
