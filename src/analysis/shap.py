from pathlib import Path
from typing import Any

import numpy as np
import shap

from src.core.logging_setup import setup_logger

logger = setup_logger("shap_service")


class SHAPService:
    """Service for computing and aggregating feature attributions."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def run_shap_analysis(
        self, model: Any, x: np.ndarray, feature_names: list[str]
    ) -> dict[str, Any]:
        """Computes SHAP values and generates state profiles."""
        logger.info("Starting SHAP analysis...")
        explainer = shap.TreeExplainer(model)
        explainer.shap_values(x)

        # Aggregate logic...
        profiles: dict[str, Any] = {"state_profiles": {}}

        self._save_profiles(profiles)
        return profiles

    def _save_profiles(self, profiles: dict[str, Any]) -> None:
        """Saves SHAP results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Save logic...
