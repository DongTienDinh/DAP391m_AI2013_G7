import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.providers.base import LLMProvider
from src.core.logging_setup import setup_logger

logger = setup_logger("xai_service")

COMP = ["PD", "GP", "PG", "MMI"]


def assign_tier(rank: int, n_states: int = 27) -> str:
    pct = rank / n_states
    if pct <= 0.19:
        return "TOP"
    if pct <= 0.38:
        return "HIGH"
    if pct <= 0.67:
        return "MID"
    return "LOW"


def compute_contributions(df: pd.DataFrame, w_star: dict[str, float], gamma: float = 0.20) -> pd.DataFrame:
    """Adds contrib, contrib_pct, dominant, weakest, flags to the dataframe."""
    df = df.copy()
    for c in COMP:
        df[f"contrib_{c}"] = w_star[c] * df[f"{c}_norm"]
        df[f"contrib_pct_{c}"] = np.where(
            df["OPP_score"] > 1e-9,
            (df[f"contrib_{c}"] / df["OPP_score"]) * 100,
            0.0,
        )

    df["risk_penalty_abs"] = df["OPP_score"] * gamma * df["LC_norm"]
    df["dominant_component"] = df[[f"contrib_{c}" for c in COMP]].idxmax(axis=1).str.replace("contrib_", "")
    df["weakest_component"] = df[[f"contrib_{c}" for c in COMP]].idxmin(axis=1).str.replace("contrib_", "")
    df["pg_saturated"] = df["PG_norm"] < 0.10
    df["high_lc_flag"] = df["LC_norm"] > 0.70
    return df


def format_narrative(exp: dict[str, Any], style: str = "brief") -> str:
    """Rule-based narrative generator."""
    state, rank, eps = exp["state"], exp["rank"], exp["eps_score"]
    dom, weak = exp["dominant_driver"], exp["weakest_component"]
    risk_pct = exp["risk_penalty_pct"]
    dom_pct = exp["components"][dom]["contrib_pct"]
    lc_norm = exp["lc_norm"]

    if style == "brief":
        return f"{state} (Rank #{rank}, EPS={eps:.1f}) — primary driver: {dom} ({dom_pct:.0f}% of OPP). Logistics penalty is {risk_pct:.1f}%."
    return (
        f"{state} ranks #{rank} with EPS={eps:.1f}. The dominant driver is {dom} ({dom_pct:.0f}% of OPP), "
        f"while {weak} is the weakest area. Logistics risk (LC_norm={lc_norm:.3f}) "
        f"penalises the opportunity score by {risk_pct:.1f}%."
    )


def build_explanation_dict(row: pd.Series, w_star: dict[str, float], gamma: float) -> dict[str, Any]:
    """Builds a rich explanation dict for one state, matching Before_Refactor's format."""
    components = {}
    for c in COMP:
        components[c] = {
            "norm": float(row[f"{c}_norm"]),
            "weight": float(w_star[c]),
            "contrib": float(row[f"contrib_{c}"]),
            "contrib_pct": float(row[f"contrib_pct_{c}"]),
            "label": c,
        }

    dominant = str(row["dominant_component"])
    weakest = str(row["weakest_component"])
    lc_norm = float(row["LC_norm"])

    exp = {
        "state": str(row["customer_state"]),
        "rank": int(row["EPS_rank"]),
        "tier": assign_tier(int(row["EPS_rank"])),
        "eps_score": float(row["EPS_score"]),
        "opp_score": float(row["OPP_score"]),
        "components": components,
        "dominant_driver": dominant,
        "weakest_component": weakest,
        "lc_norm": lc_norm,
        "risk_adj_factor": float(row["Risk_Adj"]),
        "risk_penalty_abs": float(row["risk_penalty_abs"]) if "risk_penalty_abs" in row else float(row["OPP_score"]) * gamma * lc_norm,
        "risk_penalty_pct": float(row["risk_penalty_pct"]) if "risk_penalty_pct" in row else gamma * lc_norm * 100,
        "data_sparse": bool(row.get("data_sparse", False)),
        "pg_saturated": bool(lc_norm < 0.10) if "pg_saturated" not in row else bool(row["pg_saturated"]),
        "high_lc_flag": bool(lc_norm > 0.70) if "high_lc_flag" not in row else bool(row["high_lc_flag"]),
    }
    return exp


def call_gemini_narrative(
    explanation_dict: dict[str, Any],
    national_stats: dict | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Standalone Gemini narrative call (for streamlit dynamic use). Falls back to format_narrative."""
    import os
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {"brief": format_narrative(explanation_dict, "brief"), "full": format_narrative(explanation_dict, "full")}
    try:
        from google import genai
        client = genai.Client(api_key=key)
        sp = "You are an XAI assistant. Return JSON: {'brief': '<50 words', 'full': '<150 words'}"
        user_prompt = f"Data: {json.dumps(explanation_dict)}\nNational stats: {json.dumps(national_stats or {})}"
        resp = client.models.generate_content(model="gemini-1.5-flash", contents=[sp, user_prompt])
        if resp and resp.text:
            return json.loads(resp.text.strip().strip("`json\n "))
    except Exception as e:
        logger.warning(f"call_gemini_narrative failed: {e}")
    return {"brief": format_narrative(explanation_dict, "brief"), "full": format_narrative(explanation_dict, "full")}


class XAIService:
    """Service for generating multi-tiered e-commerce market explanations."""

    def __init__(self, output_dir: Path, llm_provider: LLMProvider | None = None):
        self.output_dir = output_dir
        self.llm = llm_provider

    def run_xai_pipeline(self, eps_results_path: Path, w_star_path: Path) -> None:
        """Generates rule-based and LLM-based narratives for all states."""
        logger.info("Starting XAI Service...")

        df = pd.read_csv(eps_results_path)
        with open(w_star_path) as f:
            config = json.load(f)
        w_star = config["w_star"]
        gamma = config.get("gamma", 0.20)

        df = compute_contributions(df, w_star, gamma)
        system_prompt = (
            "You are an XAI assistant specializing in e-commerce market analytics. "
            "Generate clear, data-driven explanations. "
            "Return ONLY valid JSON: {'brief': '<under 50 words>', 'full': '<under 150 words>'}"
        )

        explanations = []
        for _, row in df.sort_values("EPS_rank").iterrows():
            exp = build_explanation_dict(row, w_star, gamma)
            exp["summary"] = format_narrative(exp, style="brief")
            exp["full_narrative"] = format_narrative(exp, style="full")

            # Try LLM if available
            llm_out = self._call_llm(exp, system_prompt)
            exp["gemini_narrative_brief"] = llm_out.get("brief", exp["summary"])
            exp["gemini_narrative_full"] = llm_out.get("full", exp["full_narrative"])

            explanations.append(exp)

        self._save_reports(explanations, w_star, gamma)
        logger.info("XAI Service completed.")

    def _call_llm(self, exp: dict[str, Any], system_prompt: str) -> dict[str, str]:
        """Calls LLM provider if available, returns fallback on failure."""
        if not self.llm or not self.llm.api_key:
            return {"brief": exp["summary"], "full": exp["full_narrative"]}
        try:
            return self.llm.generate_narrative(exp, system_prompt)
        except Exception as e:
            logger.warning(f"LLM failed for {exp['state']}: {e}")
            return {"brief": exp["summary"], "full": exp["full_narrative"]}

    def _save_reports(self, explanations: list[dict], w_star: dict, gamma: float) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": str(pd.Timestamp.now()),
            "model_config": {"gamma": gamma, "weights": w_star, "n_states": len(explanations)},
            "state_explanations": explanations,
        }
        with open(self.output_dir / "eps_xai_report.json", "w") as f:
            json.dump(report, f, indent=2)

        rows = []
        for exp in explanations:
            rows.append({
                "customer_state": exp["state"],
                "EPS_rank": exp["rank"],
                "EPS_score": exp["eps_score"],
                "tier": exp["tier"],
                "dominant_driver": exp["dominant_driver"],
                "weakest_component": exp["weakest_component"],
                "risk_penalty_pct": exp["risk_penalty_pct"],
                "data_sparse": exp["data_sparse"],
                "narrative_brief": exp["gemini_narrative_brief"],
                "narrative_full": exp["gemini_narrative_full"],
            })
        pd.DataFrame(rows).to_csv(self.output_dir / "eps_xai_report.csv", index=False)
        logger.info(f"Saved XAI reports to {self.output_dir}")
