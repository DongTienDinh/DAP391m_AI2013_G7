import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.olist_pipeline.core.logging_setup import setup_logger
from src.olist_pipeline.analysis.providers.base import LLMProvider

logger = setup_logger("xai_service")

class XAIService:
    """Service for generating multi-tiered e-commerce market explanations."""

    def __init__(self, output_dir: Path, llm_provider: Optional[LLMProvider] = None):
        self.output_dir = output_dir
        self.llm = llm_provider

    def run_xai_pipeline(self, eps_results_path: Path, w_star_path: Path) -> None:
        """Generates rule-based and LLM-based narratives for all states."""
        logger.info("Starting XAI Service...")
        
        # 1. Load EPS results
        df = pd.read_csv(eps_results_path)
        
        # 2. Generate explanations
        explanations = []
        for _, row in df.iterrows():
            exp = self._explain_state(row)
            explanations.append(exp)
            
        # 3. Save
        self._save_reports(explanations)
        logger.info("XAI Service completed.")

    def _explain_state(self, row: pd.Series) -> Dict[str, Any]:
        """Computes rule-based metrics and calls LLM if available."""
        base_info = {
            "state": row["customer_state"],
            "rank": row["EPS_rank"],
            "score": row["EPS_score"]
        }
        
        if self.llm:
            narratives = self.llm.generate_narrative(base_info, "You are an e-commerce expert...")
            base_info.update(narratives)
        else:
            base_info["brief"] = f"State {row['customer_state']} rank {row['EPS_rank']}."
            base_info["full"] = f"Detailed analysis for {row['customer_state']} showing rank {row['EPS_rank']}."
            
        return base_info

    def _save_reports(self, explanations: List[Dict[str, Any]]) -> None:
        """Saves JSON and CSV reports."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "eps_xai_report.json", "w") as f:
            json.dump({"state_explanations": explanations}, f, indent=2)
        
        pd.DataFrame(explanations).to_csv(self.output_dir / "eps_xai_report.csv", index=False)
