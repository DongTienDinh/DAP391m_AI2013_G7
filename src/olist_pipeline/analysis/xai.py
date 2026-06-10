import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.olist_pipeline.analysis.providers.base import LLMProvider
from src.olist_pipeline.core.logging_setup import setup_logger

logger = setup_logger("xai_service")


def assign_tier(rank: int, n_states: int = 27) -> str:
    """
    Assigns state rank to priority tier.
    """
    pct = rank / n_states
    if pct <= 0.19:
        return "TOP"
    if pct <= 0.38:
        return "HIGH"
    if pct <= 0.67:
        return "MID"
    return "LOW"


def format_narrative(
    explanation: Dict[str, Any], style: str = "brief", shap_context: Optional[Dict] = None
) -> str:
    """
    Rule-based narrative generator for state explanations.
    """
    state, rank, eps = explanation["state"], explanation["rank"], explanation["eps_score"]
    dom, weak = explanation["dominant_driver"], explanation["weakest_component"]
    risk_pct = explanation["risk_penalty_pct"]
    dom_pct = explanation["components"][dom]["contrib_pct"]

    if style == "brief":
        narrative = f"{state} (Rank #{rank}, EPS={eps:.1f}) — primary driver: {dom} ({dom_pct:.0f}% of OPP). Logistics penalty is {risk_pct:.1f}%."
    else:
        narrative = (
            f"{state} ranks #{rank} with EPS={eps:.1f}. The dominant driver is {dom} ({dom_pct:.0f}% of OPP), "
            f"while {weak} is the weakest area. Logistics risk (LC_norm={explanation['lc_norm']:.3f}) "
            f"penalises the opportunity score by {risk_pct:.1f}%."
        )

    if shap_context:
        verdict = shap_context.get("alignment_verdict", "N/A")
        score = shap_context.get("alignment_score", 0.0)
        narrative += f" [SHAP Alignment: {verdict} ({score:.2f})]"

    return narrative


def call_gemini_narrative(
    explanation_dict: Dict[str, Any],
    system_prompt: str,
    national_stats: Dict = None,
    api_key: Optional[str] = None,
    shap_context: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Calls the Gemini API to generate XAI narrative, fallback to rule-based.
    """
    fallback = {
        "brief": format_narrative(explanation_dict, style="brief", shap_context=shap_context),
        "full": format_narrative(explanation_dict, style="full", shap_context=shap_context),
    }

    if not api_key:
        return fallback

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        user_prompt = f"Data: {json.dumps(explanation_dict)}\nGenerate JSON: {{'brief': '...', 'full': '...'}}"

        response = client.models.generate_content(
            model="gemini-1.5-flash", contents=[system_prompt, user_prompt]
        )

        if response and response.text:
            return json.loads(response.text.strip("`json\n "))
        return fallback
    except Exception as e:
        logger.error(f"Gemini API error for {explanation_dict.get('state')}: {e}")
        return fallback


class XAIService:
    """Service for generating multi-tiered e-commerce market explanations."""

    def __init__(self, output_dir: Path, llm_provider: LLMProvider | None = None):
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

    def _explain_state(self, row: pd.Series) -> dict[str, Any]:
        """Computes rule-based metrics and calls LLM if available."""
        base_info = {
            "state": row["customer_state"],
            "rank": row["EPS_rank"],
            "score": row["EPS_score"],
        }

        if self.llm:
            narratives = self.llm.generate_narrative(base_info, "You are an e-commerce expert...")
            base_info.update(narratives)
        else:
            base_info["brief"] = f"State {row['customer_state']} rank {row['EPS_rank']}."
            base_info["full"] = (
                f"Detailed analysis for {row['customer_state']} showing rank {row['EPS_rank']}."
            )

        return base_info

    def _save_reports(self, explanations: list[dict[str, Any]]) -> None:
        """Saves JSON and CSV reports."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "eps_xai_report.json", "w") as f:
            json.dump({"state_explanations": explanations}, f, indent=2)

        pd.DataFrame(explanations).to_csv(self.output_dir / "eps_xai_report.csv", index=False)
