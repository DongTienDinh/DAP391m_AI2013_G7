#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Explainer Module — Two-Tier Feature Attribution for EPS Pipeline
"""

import sys
import io
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from src.olist_pipeline.utils.system_utils import print_section_header

# Set global random seeds for reproducibility
np.random.seed(42)

warnings.filterwarnings('ignore')

ACADEMIC_RC = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

EPS_FEATURE_MAP = {
    "PD": ["revenue_lag_1", "revenue_lag_2", "revenue_lag_4", "revenue_rolling_4", "revenue_rolling_8", "revenue_std_4", "revenue_momentum", "revenue_growth_1w", "revenue_growth_4w", "order_count", "orders_lag_1", "category_diversity", "revenue"],
    "GP": ["revenue_lag_8", "order_growth_1w"],
    "PG": ["penetration_gap", "gdp_per_capita", "population", "avg_order_value"],
    "MMI": ["unique_sellers", "seller_density", "seller_customer_ratio", "customer_seller_ratio"],
    "LC": ["avg_freight_value", "avg_delivery_time", "late_delivery_rate"],
}

EPS_COLOUR_MAP = {"PD": "#4C72B0", "GP": "#55A868", "PG": "#C44E52", "MMI": "#8172B2", "LC": "#CCB974", "Other": "#808080"}

def load_training_data(data_path: Path, config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Loads training data and prepares features.
    """
    df = pd.read_csv(data_path)
    df['state_code'] = df['customer_state'].astype('category').cat.codes
    df = df.sort_values('year_week').reset_index(drop=True)
    
    grp = df.groupby('customer_state')
    df['revenue_std_4'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(4, min_periods=2).std()).fillna(0)
    df['revenue_momentum'] = (df['revenue_lag_1'] - df['revenue_lag_4']).fillna(0)

    feature_cols = [c for c in df.columns if c not in config['feature_selection']['exclude']]
    X = df[feature_cols].values
    y_model = np.log1p(df['target_next_revenue'].values)

    return df, X, y_model, feature_cols

def train_rf_for_shap(X: np.ndarray, y_model: np.ndarray, config: Dict[str, Any]) -> RandomForestRegressor:
    """
    Trains RF for SHAP analysis.
    """
    rf_cfg = config['models']['random_forest']
    model = RandomForestRegressor(
        n_estimators=rf_cfg['n_estimators'],
        max_depth=rf_cfg['max_depth'],
        min_samples_leaf=rf_cfg['min_samples_leaf'],
        max_features=rf_cfg['max_features'],
        random_state=config['random_state'],
        n_jobs=-1,
    )
    model.fit(X, y_model)
    return model

def compute_shap_values(model: RandomForestRegressor, X: np.ndarray, feature_cols: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, pd.DataFrame(X, columns=feature_cols)

def aggregate_shap_by_state(shap_array: np.ndarray, feature_cols: List[str], df: pd.DataFrame) -> Dict[str, Dict]:
    shap_df = pd.DataFrame(np.abs(shap_array), columns=feature_cols)
    shap_df['customer_state'] = df['customer_state'].values
    state_profiles = {}
    for state, group in shap_df.groupby('customer_state'):
        mean_abs = group.drop(columns='customer_state').mean()
        ranked = mean_abs.sort_values(ascending=False)
        state_profiles[str(state)] = {
            "top_features": [{"feature": f, "mean_abs_shap": round(float(v), 6), "rank": i} for i, (f, v) in enumerate(ranked.head(10).items(), 1)],
            "total_mean_abs_shap": round(float(mean_abs.sum()), 6),
            "dominant_feature": str(ranked.index[0]),
        }
    return state_profiles

def compute_alignment_score_full(shap_array: np.ndarray, feature_cols: List[str], df: pd.DataFrame, state_profiles: Dict[str, Dict], eps_df: pd.DataFrame, w_star: Dict[str, float]) -> Dict[str, Dict]:
    shap_abs = pd.DataFrame(np.abs(shap_array), columns=feature_cols)
    shap_abs['customer_state'] = df['customer_state'].values
    
    opp_components = ["PD", "GP", "PG", "MMI"]
    w_sum = sum(w_star[c] for c in opp_components)
    w_norm = {c: w_star[c] / w_sum for c in opp_components}
    
    alignment_results = {}
    for _, row in eps_df.iterrows():
        state = str(row['customer_state'])
        if state not in state_profiles: continue
        
        group = shap_abs[shap_abs['customer_state'] == state].drop(columns='customer_state')
        feat_shap = group.mean()
        total_shap = feat_shap.sum()
        
        if total_shap < 1e-12:
            alignment_results[state] = {"overall_score": 0.0, "verdict": "LOW", "per_component": {}, "insight": "No signal"}
            continue

        comp_weights = {c: sum(feat_shap[f] for f in EPS_FEATURE_MAP[c] if f in feature_cols) / total_shap for c in opp_components}
        per_component = {}
        for c in opp_components:
            eps_w = w_star[c] * float(row.get(f'{c}_norm', 0.0)) / (float(row['OPP_score']) + 1e-9)
            per_component[c] = {"shap_weight": round(comp_weights[c], 4), "eps_weight": round(eps_w, 4), "alignment": round(1.0 - abs(comp_weights[c] - eps_w), 4)}
        
        overall = round(sum(per_component[c]['alignment'] * w_norm[c] for c in opp_components), 4)
        verdict = "HIGH" if overall >= 0.8 else ("MEDIUM" if overall >= 0.6 else "LOW")
        alignment_results[state] = {"overall_score": overall, "verdict": verdict, "per_component": per_component, "insight": f"{verdict} alignment"}
        
    return alignment_results

def plot_global_summary(shap_array, X_df, output_path):
    import shap
    with plt.rc_context(ACADEMIC_RC):
        shap.summary_plot(shap_array, X_df, show=False)
        plt.title("SHAP Feature Importance (Global)")
        plt.savefig(output_path)
        plt.close()

def plot_state_bar(state_code, profile, output_path):
    with plt.rc_context(ACADEMIC_RC):
        feats = profile['top_features']
        plt.barh([f['feature'] for f in reversed(feats)], [f['mean_abs_shap'] for f in reversed(feats)])
        plt.title(f"SHAP — {state_code}")
        plt.savefig(output_path)
        plt.close()

def load_shap_profiles(shap_path: Path) -> Dict:
    if not shap_path.exists(): return {}
    with open(shap_path, 'r') as f: return json.load(f)

def get_state_shap_context(state_code: str, profiles: Dict) -> Optional[Dict]:
    sp = profiles.get("state_profiles", {}).get(state_code)
    if not sp: return None
    return {"dominant_feature": sp["dominant_feature"], "top_3_features": [f["feature"] for f in sp["top_features"][:3]], "alignment_verdict": sp.get("alignment", {}).get("verdict", ""), "alignment_score": sp.get("alignment", {}).get("overall_score", 0.0), "alignment_insight": sp.get("alignment", {}).get("insight", "")}

def run_shap_pipeline(data_path: Path, eps_path: Path, w_star_path: Path, output_dir: Path, training_config: Dict[str, Any]) -> None:
    print_section_header("SHAP EXPLAINER PIPELINE")
    df, X, y_model, feature_cols = load_training_data(data_path, training_config)
    model = train_rf_for_shap(X, y_model, training_config)
    shap_array, X_df = compute_shap_values(model, X, feature_cols)
    state_profiles = aggregate_shap_by_state(shap_array, feature_cols, df)
    
    if eps_path.exists() and w_star_path.exists():
        eps_df = pd.read_csv(eps_path)
        with open(w_star_path, 'r') as f: w_star = json.load(f)["w_star"]
        alignments = compute_alignment_score_full(shap_array, feature_cols, df, state_profiles, eps_df, w_star)
        for s, p in state_profiles.items(): p['alignment'] = alignments.get(s, {})

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "shap_state_profiles.json", 'w') as f:
        json.dump({"metadata": {"generated_at": datetime.now().isoformat()}, "state_profiles": state_profiles}, f, indent=2)
    
    plot_global_summary(shap_array, X_df, output_dir / "shap_global_summary.png")
    for s, p in state_profiles.items(): plot_state_bar(s, p, output_dir / f"shap_state_bar_{s}.png")
    print_section_header("SHAP PIPELINE COMPLETED")
