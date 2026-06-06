#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EPS XAI Explainer Module

This module explains the decisions and metrics of the Expansion Priority Score (EPS)
pipeline for Brazilian states using rule-based metrics decomposition and the
Gemini LLM API.
"""

import os
import sys
import json
import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

from src.olist_pipeline.utils.system_utils import print_section_header
from src.olist_pipeline.utils.logger import setup_logger

logger = setup_logger("xai_explainer")

# SHAP Integration
try:
    from src.olist_pipeline.analysis.shap import load_shap_profiles, get_state_shap_context
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    def load_shap_profiles(*args, **kwargs): return {}
    def get_state_shap_context(*args, **kwargs): return None

def load_data(eps_path: Path, w_star_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads weekly features and next-week predictions.
    """
    if not eps_path.exists(): raise FileNotFoundError(f"EPS results not found: {eps_path}")
    if not w_star_path.exists(): raise FileNotFoundError(f"Weights JSON not found: {w_star_path}")
        
    logger.info(f"Loading EPS results from {eps_path.name}")
    df = pd.read_csv(eps_path)
    
    logger.info(f"Loading weights config from {w_star_path.name}")
    with open(w_star_path, 'r') as f:
        config = json.load(f)
        
    return df, config


def compute_contributions(df: pd.DataFrame, w_star: Dict[str, float], gamma: float = 0.20) -> pd.DataFrame:
    """
    Computes absolute contributions, percentage contributions, and flags.
    """
    df = df.copy()
    COMP = ['PD', 'GP', 'PG', 'MMI']
    
    for c in COMP:
        df[f'contrib_{c}'] = w_star[c] * df[f'{c}_norm']
        
    df['risk_penalty_abs'] = df['OPP_score'] * gamma * df['LC_norm']
    df['risk_penalty_pct'] = gamma * df['LC_norm'] * 100
    
    for c in COMP:
        df[f'contrib_pct_{c}'] = np.where(df['OPP_score'] > 1e-9, (df[f'contrib_{c}'] / df['OPP_score']) * 100, 0.0)
        
    df['dominant_component'] = df[[f'contrib_{c}' for c in COMP]].idxmax(axis=1).str.replace('contrib_', '')
    df['weakest_component'] = df[[f'contrib_{c}' for c in COMP]].idxmin(axis=1).str.replace('contrib_', '')
    
    df['pg_saturated'] = df['PG_norm'] < 0.10
    df['high_lc_flag'] = df['LC_norm'] > 0.70
    
    # MMI imputed check (heuristic fallback)
    mmi_median = df['MMI_norm'].median()
    df['mmi_imputed'] = np.abs(df['MMI_norm'] - mmi_median) < 1e-4
    
    return df


def assign_tier(rank: int, n_states: int = 27) -> str:
    """
    Assigns state rank to priority tier.
    """
    pct = rank / n_states
    if pct <= 0.19: return "TOP"
    if pct <= 0.38: return "HIGH"
    if pct <= 0.67: return "MID"
    return "LOW"


def explain_contrastive(state_code: str, df: pd.DataFrame, w_star: Dict[str, float], gamma: float) -> Dict[str, Any]:
    """Compare target state against neighbors in ranking."""
    target_row = df[df['customer_state'] == state_code].iloc[0]
    rank = int(target_row['EPS_rank'])
    comps = ['PD', 'GP', 'PG', 'MMI']
    
    def _compare(r_target, r_other, is_above: bool):
        if r_other.empty: return None
        r_other = r_other.iloc[0]
        other_state = r_other['customer_state']
        diffs = {c: float(r_target[f'contrib_{c}'] - r_other[f'contrib_{c}']) for c in comps}
        lc_adv = float(r_other['LC_norm'] - r_target['LC_norm'])
        
        if is_above:
            worst_comp = min(diffs, key=diffs.get)
            msg = f"falls behind {other_state} primarily because {worst_comp} contribution is lower by {abs(diffs[worst_comp]):.3f}"
            primary = worst_comp
        else:
            best_comp = max(diffs, key=diffs.get)
            msg = f"ranks above {other_state} primarily because {best_comp} contribution exceeds {other_state} by {diffs[best_comp]:.3f}"
            primary = best_comp
        return {"state": other_state, "primary_component": primary, "narrative": msg}

    vs_above = _compare(target_row, df[df['EPS_rank'] == rank - 1], True)
    vs_below = _compare(target_row, df[df['EPS_rank'] == rank + 1], False)
    
    narrative = []
    if vs_above: narrative.append(f"{state_code} {vs_above['narrative']}.")
    if vs_below: narrative.append(f"{state_code} {vs_below['narrative']}.")
    
    return {"vs_above": vs_above, "vs_below": vs_below, "narrative_contrastive": " ".join(narrative)}


def format_narrative(explanation: Dict[str, Any], style: str = 'brief', shap_context: Optional[Dict] = None) -> str:
    """
    Rule-based narrative generator for state explanations.
    """
    state, rank, eps = explanation['state'], explanation['rank'], explanation['eps_score']
    dom, weak = explanation['dominant_driver'], explanation['weakest_component']
    risk_pct = explanation['risk_penalty_pct']
    dom_pct = explanation['components'][dom]['contrib_pct']

    if style == 'brief':
        narrative = f"{state} (Rank #{rank}, EPS={eps:.1f}) — primary driver: {dom} ({dom_pct:.0f}% of OPP). Logistics penalty is {risk_pct:.1f}%."
    else:
        narrative = (f"{state} ranks #{rank} with EPS={eps:.1f}. The dominant driver is {dom} ({dom_pct:.0f}% of OPP), "
                     f"while {weak} is the weakest area. Logistics risk (LC_norm={explanation['lc_norm']:.3f}) "
                     f"penalises the opportunity score by {risk_pct:.1f}%.")
        
    if shap_context:
        verdict = shap_context.get('alignment_verdict', 'N/A')
        score = shap_context.get('alignment_score', 0.0)
        narrative += f" [SHAP Alignment: {verdict} ({score:.2f})]"

    return narrative


def call_gemini_narrative(explanation_dict: Dict[str, Any], system_prompt: str, national_stats: Dict = None, api_key: Optional[str] = None, shap_context: Optional[Dict] = None) -> Dict[str, str]:
    """
    Calls the Gemini API to generate XAI narrative, fallback to rule-based.
    """
    fallback = {
        "brief": format_narrative(explanation_dict, style='brief', shap_context=shap_context),
        "full": format_narrative(explanation_dict, style='full', shap_context=shap_context)
    }
    
    if not api_key: return fallback
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        user_prompt = f"Data: {json.dumps(explanation_dict)}\nGenerate JSON: {{'brief': '...', 'full': '...'}}"
        
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[system_prompt, user_prompt]
        )
        
        if response and response.text:
            return json.loads(response.text.strip('`json\n '))
        return fallback
    except Exception as e:
        logger.error(f"Gemini API error for {explanation_dict.get('state')}: {e}")
        return fallback


def explain_state(state_code: str, df_contrib: pd.DataFrame, w_star: Dict[str, float], system_prompt: str, national_stats: Dict = None, gamma: float = 0.20, api_key: Optional[str] = None, shap_profiles: Dict = None) -> Dict[str, Any]:
    """
    Computes XAI fields and structured explanations for a single state.
    """
    row = df_contrib[df_contrib['customer_state'] == state_code].iloc[0]
    
    explanation = {
        "state": state_code,
        "rank": int(row['EPS_rank']),
        "tier": assign_tier(int(row['EPS_rank']), len(df_contrib)),
        "eps_score": float(row['EPS_score']),
        "opp_score": float(row['OPP_score']),
        "components": {c: {"norm": float(row[f'{c}_norm']), "weight": float(w_star[c]), "contrib_pct": float(row[f'contrib_pct_{c}'])} for c in ['PD', 'GP', 'PG', 'MMI']},
        "dominant_driver": str(row['dominant_component']),
        "weakest_component": str(row['weakest_component']),
        "lc_norm": float(row['LC_norm']),
        "risk_penalty_pct": float(row['risk_penalty_pct']),
        "data_sparse": bool(row['data_sparse']),
        "pg_saturated": bool(row['pg_saturated']),
        "high_lc_flag": bool(row['high_lc_flag']),
        "mmi_imputed": bool(row['mmi_imputed'])
    }
    
    explanation["contrastive"] = explain_contrastive(state_code, df_contrib, w_star, gamma)
    shap_ctx = get_state_shap_context(state_code, shap_profiles or {})
    explanation["shap_context"] = shap_ctx or {}
    
    gemini_out = call_gemini_narrative(explanation, system_prompt, national_stats, api_key=api_key, shap_context=shap_ctx)
    explanation["gemini_narrative_brief"] = gemini_out.get("brief", "")
    explanation["gemini_narrative_full"] = gemini_out.get("full", "")
    
    return explanation

def run_xai_pipeline(eps_path: Path, w_star_path: Path, shap_profiles_path: Path, output_dir: Path, api_key: Optional[str], config: Dict[str, Any]) -> None:
    """
    Executes the full XAI explainer pipeline.
    """
    print_section_header("RUNNING EPS XAI EXPLAINER PIPELINE")
    
    df, w_config = load_data(eps_path, w_star_path)
    
    df_contrib = compute_contributions(df, w_config["w_star"], w_config.get("gamma", 0.20))
    shap_profiles = load_shap_profiles(shap_profiles_path)
    
    explanations = []
    for i, row in df_contrib.sort_values('EPS_rank').iterrows():
        state_code = row['customer_state']
        logger.info(f"[{i+1}/27] Explaining state: {state_code}")
        
        exp = explain_state(state_code, df_contrib, w_config["w_star"], config['xai']['system_prompt'], gamma=w_config.get("gamma", 0.20), api_key=api_key, shap_profiles=shap_profiles)
        explanations.append(exp)
        
        if api_key: time.sleep(15) # Rate limit

    # Save reports
    with open(output_dir / "eps_xai_report.json", 'w') as f:
        json.dump({"state_explanations": explanations}, f, indent=2)
    
    # Build and save CSV Report
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
            "narrative_brief": exp["gemini_narrative_brief"] or format_narrative(exp, 'brief'),
            "narrative_full": exp["gemini_narrative_full"] or format_narrative(exp, 'full')
        })
    pd.DataFrame(rows).to_csv(output_dir / "eps_xai_report.csv", index=False)
    
    print_section_header("XAI PIPELINE COMPLETED")
