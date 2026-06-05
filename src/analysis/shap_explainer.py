#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Explainer Module — Two-Tier Feature Attribution for EPS Pipeline

Tier 1: Global SHAP feature importance (national RF predictions)
Tier 2: State-level SHAP aggregation with SHAP-EPS Alignment Score

The SHAP-EPS Alignment Score validates whether features driving RF predictions
for a state are consistent with that state's EPS ranking — detecting states
that are underranked due to data quality vs. genuinely low potential.

Usage:
    python src/analysis/shap_explainer.py [--eps-path ...] [--output-dir ...]
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

# Set global random seeds for reproducibility
np.random.seed(42)

# Force UTF-8 stdout/stderr encoding for safe terminal logs on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Headless Matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# Add project root to PYTHONPATH for standalone execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.system_utils import print_section_header

# ── Academic Plotting Style (identical to train_model.py) ────────────────────

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

# ── EPS Component → Training Feature Mapping ─────────────────────────────────
#
# Maps each EPS component to the RF training features that conceptually
# drive that component.  Only features surviving select_features() are listed.

EPS_FEATURE_MAP = {
    "PD": [
        # Direct revenue lag/rolling signals → Predicted Demand
        "revenue_lag_1", "revenue_lag_2", "revenue_lag_4",
        "revenue_rolling_4", "revenue_rolling_8",
        "revenue_std_4", "revenue_momentum",
        "revenue_growth_1w", "revenue_growth_4w",
        # Order volume proxies for demand
        "order_count", "orders_lag_1",
        # Category diversity drives demand breadth
        "category_diversity",
        # Revenue itself as a demand signal
        "revenue",
    ],
    "GP": [
        # Longer-lag revenue for growth trend computation
        "revenue_lag_8",
        "order_growth_1w",
    ],
    "PG": [
        # Penetration gap and market size features
        "penetration_gap", "gdp_per_capita",
        # Population drives expected revenue denominator in PG formula
        "population",
        # Average order value reflects purchasing depth
        "avg_order_value",
    ],
    "MMI": [
        # Seller-side momentum features
        "unique_sellers", "seller_density",
        "seller_customer_ratio", "customer_seller_ratio",
    ],
    "LC": [
        # Logistics cost and delivery quality features
        "avg_freight_value", "avg_delivery_time", "late_delivery_rate",
    ],
}

# Colour palette for EPS component groups in bar charts
EPS_COLOUR_MAP = {
    "PD":    "#4C72B0",
    "GP":    "#55A868",
    "PG":    "#C44E52",
    "MMI":   "#8172B2",
    "LC":    "#CCB974",
    "Other": "#808080",
}


# ── Data Loading (replicates train_model.py pipeline exactly) ────────────────

def load_training_data(project_root: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Replicates train_model.py pipeline exactly to get identical X and feature_cols.

    Steps: load_and_prepare_data → add_dynamic_features → select_features(log_target=True)

    Returns:
        df:           Full DataFrame after preparation and dynamic features
        X:            numpy array of shape (n_samples, n_features)
        y_model:      log1p-transformed target (for RF training)
        feature_cols: list of feature column names
    """
    data_path = project_root / "data" / "processed" / "olist" / "features_weekly.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    print(f"  Loading training data from {data_path.name}...")
    df = pd.read_csv(data_path)

    # ── load_and_prepare_data ──
    df['state_code'] = df['customer_state'].astype('category').cat.codes
    df = df.sort_values('year_week').reset_index(drop=True)

    # ── add_dynamic_features ──
    grp = df.groupby('customer_state')
    df['revenue_std_4'] = grp['revenue'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std()
    ).fillna(0)
    df['revenue_momentum'] = (df['revenue_lag_1'] - df['revenue_lag_4']).fillna(0)

    # ── select_features (log_target=True) ──
    exclude = [
        'customer_state', 'year_week', 'target_next_revenue',
        'sales_per_capita', 'orders_per_capita',
        # Remove columns with high multicollinearity or data leakage
        'payment_value',           # corr=1.0000 with revenue
        'unique_customers',        # corr=1.0000 with order_count
        'item_count',              # corr=0.9994 with order_count
        'customers_lag_1',         # corr=1.0000 with orders_lag_1
        'purchasing_power_index',  # corr=1.0000 with gdp_per_capita
        'customer_penetration',    # corr=1.0000 with penetration_gap
        'revenue_ewm_4',           # corr=0.99 with rolling features
        'revenue_ewm_8',           # corr=0.99 with rolling features
        'revenue_rolling_12',      # corr=0.99 with rolling_8
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df['target_next_revenue'].values
    y_model = np.log1p(y)

    print(f"  Loaded {df.shape[0]:,} samples × {len(feature_cols)} features")
    return df, X, y_model, feature_cols


# ── RF Model Training ────────────────────────────────────────────────────────

def train_rf_for_shap(X: np.ndarray, y_model: np.ndarray) -> RandomForestRegressor:
    """
    Trains RandomForestRegressor with identical hyperparameters as train_model.py.

    Fits on full dataset (no CV split — SHAP needs stable global model).
    """
    print("  Training Random Forest on full dataset for SHAP...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y_model)
    print(f"  RF fitted — OOB not used (full-data fit for SHAP stability)")
    return model


# ── SHAP Computation ─────────────────────────────────────────────────────────

def compute_shap_values(
    model: RandomForestRegressor,
    X: np.ndarray,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Uses shap.TreeExplainer(model).

    Returns:
        shap_array: shape (n_samples, n_features) — raw SHAP values
        X_df:       pd.DataFrame with feature_cols as columns (for shap plots)
    """
    import shap

    print("  Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    X_df = pd.DataFrame(X, columns=feature_cols)
    print(f"  SHAP values computed — shape {shap_values.shape}")
    return shap_values, X_df


# ── State-Level Aggregation ──────────────────────────────────────────────────

def aggregate_shap_by_state(
    shap_array: np.ndarray,
    feature_cols: List[str],
    df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Groups rows by customer_state, computes mean absolute SHAP per feature per state.

    Returns dict keyed by state code with top_features, total_mean_abs_shap,
    and dominant_feature.
    """
    print("  Aggregating SHAP values by state...")
    shap_df = pd.DataFrame(np.abs(shap_array), columns=feature_cols)
    shap_df['customer_state'] = df['customer_state'].values

    state_profiles: Dict[str, Dict] = {}

    for state, group in shap_df.groupby('customer_state'):
        # Mean absolute SHAP per feature for this state
        mean_abs = group.drop(columns='customer_state').mean()
        total_mean_abs = float(mean_abs.sum())

        # Rank features by mean absolute SHAP
        ranked = mean_abs.sort_values(ascending=False)
        top_features = []
        for rank_idx, (feat, val) in enumerate(ranked.head(10).items(), start=1):
            top_features.append({
                "feature": feat,
                "mean_abs_shap": round(float(val), 6),
                "rank": rank_idx,
            })

        state_profiles[str(state)] = {
            "top_features": top_features,
            "total_mean_abs_shap": round(total_mean_abs, 6),
            "dominant_feature": str(ranked.index[0]),
        }

    print(f"  Aggregated profiles for {len(state_profiles)} states")
    return state_profiles


# ── SHAP-EPS Alignment Score ─────────────────────────────────────────────────

def _feature_to_eps_component(feature: str) -> str:
    """Maps a single feature name to its EPS component group, or 'Other'."""
    for comp, features in EPS_FEATURE_MAP.items():
        if feature in features:
            return comp
    return "Other"


def compute_alignment_score(
    state_shap_profiles: Dict[str, Dict],
    eps_df: pd.DataFrame,
    w_star: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Computes SHAP-EPS Alignment Score for each state.

    For each EPS component C (PD, GP, PG, MMI):
      - component_shap_weight(C) = sum of mean_abs_shap for features in EPS_FEATURE_MAP[C]
                                   / total_mean_abs_shap for that state
      - component_eps_weight(C)  = w_star[C] * norm_C / OPP_score for that state

    alignment_per_component(C) = 1 - |component_shap_weight(C) - component_eps_weight(C)|
    overall_alignment = weighted average of alignment_per_component (weights = w_star)
    """
    print("  Computing SHAP-EPS Alignment Scores...")

    # Normalise w_star to sum=1 for weighting
    opp_components = ["PD", "GP", "PG", "MMI"]
    w_sum = sum(w_star[c] for c in opp_components)
    w_norm = {c: w_star[c] / w_sum for c in opp_components}

    alignment_results: Dict[str, Dict] = {}

    for _, row in eps_df.iterrows():
        state = str(row['customer_state'])
        if state not in state_shap_profiles:
            continue

        profile = state_shap_profiles[state]
        total_shap = profile['total_mean_abs_shap']

        # Avoid division by zero
        if total_shap < 1e-12:
            alignment_results[state] = {
                "overall_score": 0.0,
                "verdict": "LOW",
                "per_component": {},
                "insight": f"No SHAP signal for {state} — total mean |SHAP| ≈ 0.",
            }
            continue

        opp_score = float(row['OPP_score'])

        # Build a lookup: feature → mean_abs_shap from top_features + all features
        # We need ALL features' SHAP, not just top 10.
        # Reconstruct from the shap array indirectly via stored total and top features.
        # Actually, top_features only has top 10. For alignment we need all features
        # mapped to components. We can approximate using the top_features list
        # and assign remaining SHAP to "Other".
        # Better: store all features' mean_abs_shap during aggregation. But the spec
        # says top 10. We'll compute component weights from the full shap data that
        # was passed to the aggregation function.
        #
        # Since we stored total_mean_abs_shap, and the profile only has top 10,
        # we need the full per-feature data. We'll handle this by passing the full
        # shap data separately. For now, use the top_features as a reasonable
        # approximation — features outside top 10 typically have negligible SHAP.

        # Sum SHAP weights by component from top features
        component_shap_sums: Dict[str, float] = {c: 0.0 for c in opp_components}
        for feat_info in profile['top_features']:
            comp = _feature_to_eps_component(feat_info['feature'])
            if comp in opp_components:
                component_shap_sums[comp] += feat_info['mean_abs_shap']

        per_component: Dict[str, Dict] = {}
        alignment_values: List[float] = []
        weights_list: List[float] = []

        for c in opp_components:
            shap_weight = component_shap_sums[c] / total_shap

            # EPS weight: w_star[C] * norm_C / OPP_score
            norm_val = float(row.get(f'{c}_norm', 0.0))
            if opp_score > 1e-9:
                eps_weight = w_star[c] * norm_val / opp_score
            else:
                eps_weight = 0.0

            comp_alignment = 1.0 - abs(shap_weight - eps_weight)
            comp_alignment = max(0.0, min(1.0, comp_alignment))  # clamp [0, 1]

            per_component[c] = {
                "shap_weight": round(shap_weight, 4),
                "eps_weight": round(eps_weight, 4),
                "alignment": round(comp_alignment, 4),
            }

            alignment_values.append(comp_alignment)
            weights_list.append(w_norm[c])

        # Weighted average alignment
        overall = sum(a * w for a, w in zip(alignment_values, weights_list))
        overall = round(overall, 4)

        # Verdict
        if overall >= 0.80:
            verdict = "HIGH"
        elif overall >= 0.60:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"

        # Generate insight
        dominant_feature = profile['dominant_feature']
        data_sparse = bool(row.get('data_sparse', False))

        # Find dominant EPS component (highest contribution)
        dominant_eps_comp = max(
            opp_components,
            key=lambda c: w_star[c] * float(row.get(f'{c}_norm', 0.0))
        )

        if verdict == "HIGH":
            insight = (
                f"SHAP drivers align with EPS ranking — {dominant_feature} consistently "
                f"supports the {dominant_eps_comp} signal."
            )
        elif verdict == "MEDIUM":
            insight = (
                f"Moderate alignment — {dominant_feature} is the strongest RF predictor "
                f"for {state}, partially consistent with EPS component weights. "
                f"Some divergence in {min(per_component, key=lambda c: per_component[c]['alignment'])} "
                f"component suggests the model captures signals beyond the EPS formula."
            )
        elif verdict == "LOW" and data_sparse:
            insight = (
                f"Low alignment likely reflects data sparsity — RF extrapolates "
                f"from national patterns rather than {state}-specific signals."
            )
        else:  # LOW, not sparse
            insight = (
                f"Low alignment — {dominant_feature} drives RF predictions for {state} "
                f"but is weakly represented in EPS components. "
                f"Consider whether {dominant_feature} should inform EPS formula revision."
            )

        # Override insight for extreme component-level misalignment
        worst_comp = min(per_component, key=lambda c: per_component[c]['alignment'])
        worst_alignment = per_component[worst_comp]['alignment']
        worst_shap = per_component[worst_comp]['shap_weight']
        worst_eps = per_component[worst_comp]['eps_weight']

        if worst_alignment < 0.45 and worst_eps > 0.5:
            # EPS heavily weights a component that RF barely sees
            insight = (
                f"Critical misalignment on {worst_comp} component "
                f"(EPS weight={worst_eps:.2f}, SHAP weight={worst_shap:.2f}) — "
                f"EPS assigns high importance to {worst_comp} but RF finds negligible "
                f"signal for {state}. This state may be overranked on {worst_comp} "
                f"due to {'data sparsity' if data_sparse else 'a model-formula gap'}."
            )
        elif worst_alignment < 0.25 and worst_shap > 0.5:
            # RF heavily weights a component that EPS undervalues
            insight = (
                f"Critical misalignment on {worst_comp} component "
                f"(SHAP weight={worst_shap:.2f}, EPS weight={worst_eps:.2f}) — "
                f"RF identifies {worst_comp}-related features as dominant for {state} "
                f"but EPS formula underweights this signal. "
                f"Consider revising EPS component weights for this state tier."
            )

        alignment_results[state] = {
            "overall_score": overall,
            "verdict": verdict,
            "per_component": per_component,
            "insight": insight,
        }

    print(f"  Alignment scores computed for {len(alignment_results)} states")
    return alignment_results


# ── Full-Feature State Aggregation (for accurate alignment computation) ──────

def _aggregate_shap_full(
    shap_array: np.ndarray,
    feature_cols: List[str],
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-state, per-feature mean absolute SHAP values (ALL features).
    Used internally for accurate alignment computation.
    """
    shap_abs = pd.DataFrame(np.abs(shap_array), columns=feature_cols)
    shap_abs['customer_state'] = df['customer_state'].values

    result: Dict[str, Dict[str, float]] = {}
    for state, group in shap_abs.groupby('customer_state'):
        means = group.drop(columns='customer_state').mean()
        result[str(state)] = {feat: float(val) for feat, val in means.items()}
    return result


def compute_alignment_score_full(
    full_shap_by_state: Dict[str, Dict[str, float]],
    state_shap_profiles: Dict[str, Dict],
    eps_df: pd.DataFrame,
    w_star: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Enhanced alignment score using full per-feature SHAP (not just top 10).
    """
    print("  Computing SHAP-EPS Alignment Scores (full feature set)...")

    opp_components = ["PD", "GP", "PG", "MMI"]
    w_sum = sum(w_star[c] for c in opp_components)
    w_norm = {c: w_star[c] / w_sum for c in opp_components}

    alignment_results: Dict[str, Dict] = {}

    for _, row in eps_df.iterrows():
        state = str(row['customer_state'])
        if state not in full_shap_by_state or state not in state_shap_profiles:
            continue

        feat_shap = full_shap_by_state[state]
        profile = state_shap_profiles[state]
        total_shap = sum(feat_shap.values())

        if total_shap < 1e-12:
            alignment_results[state] = {
                "overall_score": 0.0,
                "verdict": "LOW",
                "per_component": {},
                "insight": f"No SHAP signal for {state} — total mean |SHAP| ≈ 0.",
            }
            continue

        opp_score = float(row['OPP_score'])

        # Component SHAP weights from ALL features
        component_shap_sums: Dict[str, float] = {c: 0.0 for c in opp_components}
        for feat_name, shap_val in feat_shap.items():
            comp = _feature_to_eps_component(feat_name)
            if comp in opp_components:
                component_shap_sums[comp] += shap_val

        per_component: Dict[str, Dict] = {}
        alignment_values: List[float] = []
        weights_list: List[float] = []

        for c in opp_components:
            shap_weight = component_shap_sums[c] / total_shap

            norm_val = float(row.get(f'{c}_norm', 0.0))
            if opp_score > 1e-9:
                eps_weight = w_star[c] * norm_val / opp_score
            else:
                eps_weight = 0.0

            comp_alignment = 1.0 - abs(shap_weight - eps_weight)
            comp_alignment = max(0.0, min(1.0, comp_alignment))

            per_component[c] = {
                "shap_weight": round(shap_weight, 4),
                "eps_weight": round(eps_weight, 4),
                "alignment": round(comp_alignment, 4),
            }

            alignment_values.append(comp_alignment)
            weights_list.append(w_norm[c])

        overall = round(sum(a * w for a, w in zip(alignment_values, weights_list)), 4)

        if overall >= 0.80:
            verdict = "HIGH"
        elif overall >= 0.60:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"

        dominant_feature = profile['dominant_feature']
        data_sparse = bool(row.get('data_sparse', False))

        dominant_eps_comp = max(
            opp_components,
            key=lambda c: w_star[c] * float(row.get(f'{c}_norm', 0.0))
        )

        if verdict == "HIGH":
            insight = (
                f"SHAP drivers align with EPS ranking — {dominant_feature} consistently "
                f"supports the {dominant_eps_comp} signal."
            )
        elif verdict == "MEDIUM":
            insight = (
                f"Moderate alignment — {dominant_feature} is the strongest RF predictor "
                f"for {state}, partially consistent with EPS component weights. "
                f"Some divergence in {min(per_component, key=lambda c: per_component[c]['alignment'])} "
                f"component suggests the model captures signals beyond the EPS formula."
            )
        elif verdict == "LOW" and data_sparse:
            insight = (
                f"Low alignment likely reflects data sparsity — RF extrapolates "
                f"from national patterns rather than {state}-specific signals."
            )
        else:  # LOW, not sparse
            insight = (
                f"Low alignment — {dominant_feature} drives RF predictions for {state} "
                f"but is weakly represented in EPS components. "
                f"Consider whether {dominant_feature} should inform EPS formula revision."
            )

        # Override insight for extreme component-level misalignment
        worst_comp = min(per_component, key=lambda c: per_component[c]['alignment'])
        worst_alignment = per_component[worst_comp]['alignment']
        worst_shap = per_component[worst_comp]['shap_weight']
        worst_eps = per_component[worst_comp]['eps_weight']

        if worst_alignment < 0.45 and worst_eps > 0.5:
            # EPS heavily weights a component that RF barely sees
            insight = (
                f"Critical misalignment on {worst_comp} component "
                f"(EPS weight={worst_eps:.2f}, SHAP weight={worst_shap:.2f}) — "
                f"EPS assigns high importance to {worst_comp} but RF finds negligible "
                f"signal for {state}. This state may be overranked on {worst_comp} "
                f"due to {'data sparsity' if data_sparse else 'a model-formula gap'}."
            )
        elif worst_alignment < 0.25 and worst_shap > 0.5:
            # RF heavily weights a component that EPS undervalues
            insight = (
                f"Critical misalignment on {worst_comp} component "
                f"(SHAP weight={worst_shap:.2f}, EPS weight={worst_eps:.2f}) — "
                f"RF identifies {worst_comp}-related features as dominant for {state} "
                f"but EPS formula underweights this signal. "
                f"Consider revising EPS component weights for this state tier."
            )

        alignment_results[state] = {
            "overall_score": overall,
            "verdict": verdict,
            "per_component": per_component,
            "insight": insight,
        }

    print(f"  Alignment scores computed for {len(alignment_results)} states")
    return alignment_results


# ── Visualisation Functions ──────────────────────────────────────────────────

def plot_global_summary(
    shap_array: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Academic-quality SHAP dot plot (global).
    Title: "SHAP Feature Importance — Random Forest (Global)".
    Saved as PNG at 300 dpi.
    """
    import shap

    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_array,
            X_df,
            max_display=top_n,
            show=False,
            plot_size=None,
        )
        ax = plt.gca()
        ax.set_title("SHAP Feature Importance — Random Forest (Global)",
                      fontweight="bold")
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(str(output_path), format="png")
        plt.close(fig)
    print(f"  Saved global SHAP summary → {output_path.name}")


def plot_state_bar(
    state_code: str,
    state_profile: Dict,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Horizontal bar chart of mean absolute SHAP values for top features.
    Bars coloured by EPS component group using EPS_FEATURE_MAP.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        features = state_profile['top_features'][:top_n]
        names = [f['feature'] for f in reversed(features)]
        values = [f['mean_abs_shap'] for f in reversed(features)]
        colours = [EPS_COLOUR_MAP.get(_feature_to_eps_component(n), "#808080")
                    for n in names]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(names, values, color=colours, edgecolor="white", linewidth=0.6)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Attribution — {state_code}", fontweight="bold")

        # Legend for component colours
        legend_handles = [
            mpatches.Patch(color=col, label=comp)
            for comp, col in EPS_COLOUR_MAP.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            framealpha=0.9,
            title="EPS Component",
            fontsize=7,
            title_fontsize=8,
        )

        fig.tight_layout()
        fig.savefig(str(output_path), format="png")
        plt.close(fig)


def plot_alignment_heatmap(
    alignment_results: Dict[str, Dict],
    eps_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Heatmap: states (rows, sorted by EPS rank) × EPS components (columns).
    Cell values = per-component alignment score (0–1).
    Right-side column = overall alignment verdict.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        eps_sorted = eps_df.sort_values('EPS_rank')
        opp_components = ["PD", "GP", "PG", "MMI"]

        states_ordered = []
        matrix_rows = []
        verdicts = []
        overall_scores = []

        for _, row in eps_sorted.iterrows():
            state = str(row['customer_state'])
            if state not in alignment_results:
                continue

            ar = alignment_results[state]
            states_ordered.append(f"{state} (#{int(row['EPS_rank'])})")

            comp_vals = []
            for c in opp_components:
                val = ar.get('per_component', {}).get(c, {}).get('alignment', 0.0)
                comp_vals.append(val)
            matrix_rows.append(comp_vals)
            verdicts.append(ar.get('verdict', '?'))
            overall_scores.append(ar.get('overall_score', 0.0))

        if not matrix_rows:
            print("  [Warning] No alignment data for heatmap — skipping.")
            return

        matrix = np.array(matrix_rows)

        fig, (ax_heat, ax_verdict) = plt.subplots(
            1, 2, figsize=(10, max(6, len(states_ordered) * 0.35)),
            gridspec_kw={"width_ratios": [4, 1]},
            sharey=True,
        )

        # Main heatmap
        sns.heatmap(
            matrix,
            ax=ax_heat,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=opp_components,
            yticklabels=states_ordered,
            linewidths=0.5,
            cbar_kws={"label": "Alignment Score", "shrink": 0.6},
        )
        ax_heat.set_title("SHAP–EPS Alignment Score by State and Component",
                          fontweight="bold", pad=12)
        ax_heat.set_xlabel("")

        # Verdict sidebar
        verdict_colours = {"HIGH": "#2ca02c", "MEDIUM": "#ff7f0e", "LOW": "#d62728"}
        for i, (verdict, score) in enumerate(zip(verdicts, overall_scores)):
            colour = verdict_colours.get(verdict, "#808080")
            ax_verdict.barh(i, 1, color=colour, edgecolor="white", linewidth=0.5)
            ax_verdict.text(0.5, i, f"{verdict}\n{score:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold",
                            color="white")

        ax_verdict.set_xlim(0, 1)
        ax_verdict.set_xticks([])
        ax_verdict.set_title("Verdict", fontweight="bold", pad=12)
        ax_verdict.invert_yaxis()

        fig.tight_layout()
        fig.savefig(str(output_path), format="png")
        plt.close(fig)
    print(f"  Saved alignment heatmap → {output_path.name}")


# ── Public API (for import by eps_xai_explainer.py) ──────────────────────────

def load_shap_profiles(
    shap_path: str = "outputs/eps/shap/shap_state_profiles.json",
) -> Dict:
    """Loads pre-computed shap_state_profiles.json. Returns empty dict if not found."""
    p = Path(shap_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p

    if not p.exists():
        return {}

    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_state_shap_context(state_code: str, profiles: Dict) -> Optional[Dict]:
    """
    Returns a flat dict for easy narrative injection into eps_xai_explainer.

    Keys: dominant_feature, top_3_features, alignment_verdict,
          alignment_score, alignment_insight.

    Returns None if state not found in profiles.
    """
    state_profiles = profiles.get("state_profiles", {})
    if state_code not in state_profiles:
        return None

    sp = state_profiles[state_code]
    top_features = sp.get("top_features", [])
    alignment = sp.get("alignment", {})

    return {
        "dominant_feature": sp.get("dominant_feature", ""),
        "top_3_features": [f["feature"] for f in top_features[:3]],
        "alignment_verdict": alignment.get("verdict", ""),
        "alignment_score": alignment.get("overall_score", 0.0),
        "alignment_insight": alignment.get("insight", ""),
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP Explainer — Two-tier feature attribution for EPS pipeline."
    )
    parser.add_argument(
        "--eps-path",
        type=str,
        default="outputs/eps/eps_results.csv",
        help="Path to EPS results CSV",
    )
    parser.add_argument(
        "--w-star-path",
        type=str,
        default="outputs/eps/w_star.json",
        help="Path to w_star weight configuration JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eps/shap",
        help="Directory to save SHAP outputs",
    )
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[2]
    output_dir = proj_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eps_path = proj_root / args.eps_path if not Path(args.eps_path).is_absolute() else Path(args.eps_path)
    w_star_path = proj_root / args.w_star_path if not Path(args.w_star_path).is_absolute() else Path(args.w_star_path)

    # ── 1. Load training data ──
    print_section_header("SHAP EXPLAINER — TIER 1: DATA & MODEL")
    df, X, y_model, feature_cols = load_training_data(proj_root)

    # ── 2. Train RF on full dataset ──
    model = train_rf_for_shap(X, y_model)

    # ── 3. Compute SHAP values ──
    shap_array, X_df = compute_shap_values(model, X, feature_cols)

    # ── 4. Aggregate SHAP by state ──
    print_section_header("SHAP EXPLAINER — TIER 2: STATE PROFILES")
    state_profiles = aggregate_shap_by_state(shap_array, feature_cols, df)

    # Full per-feature aggregation for accurate alignment
    full_shap_by_state = _aggregate_shap_full(shap_array, feature_cols, df)

    # ── 5. Load EPS results + w_star (graceful if missing) ──
    eps_df = None
    w_star = None
    gamma = 0.20
    alignment_results: Dict[str, Dict] = {}

    if eps_path.exists() and w_star_path.exists():
        print(f"  Loading EPS results from {eps_path.name}...")
        eps_df = pd.read_csv(eps_path)

        with open(w_star_path, 'r') as f:
            config = json.load(f)
        w_star = config["w_star"]
        gamma = config.get("gamma", 0.20)

        # ── 6. Compute alignment scores ──
        alignment_results = compute_alignment_score_full(
            full_shap_by_state, state_profiles, eps_df, w_star,
        )
    else:
        warnings.warn(
            f"EPS files not found ({eps_path.name}, {w_star_path.name}). "
            f"Skipping alignment score computation.",
            stacklevel=2,
        )

    # ── 7. Merge alignment into state profiles ──
    for state, profile in state_profiles.items():
        if state in alignment_results:
            profile['alignment'] = alignment_results[state]

    # ── 8. Build and save shap_state_profiles.json ──
    print_section_header("SAVING SHAP OUTPUTS")

    # Global feature importance
    mean_abs_global = np.abs(shap_array).mean(axis=0)
    global_ranking = np.argsort(-mean_abs_global)
    global_importance = []
    for rank_idx, feat_idx in enumerate(global_ranking, start=1):
        global_importance.append({
            "feature": feature_cols[feat_idx],
            "mean_abs_shap": round(float(mean_abs_global[feat_idx]), 6),
            "rank": rank_idx,
        })

    output_json = {
        "metadata": {
            "model": "Random Forest",
            "n_estimators": 300,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "generated_at": datetime.now().isoformat(),
            "shap_method": "TreeExplainer",
        },
        "global_feature_importance": global_importance,
        "state_profiles": state_profiles,
    }

    json_path = output_dir / "shap_state_profiles.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {json_path.name}")

    # ── 9. Generate all plots ──
    print_section_header("GENERATING SHAP VISUALISATIONS")

    # 9a. Global summary dot plot
    plot_global_summary(
        shap_array, X_df,
        output_path=output_dir / "shap_global_summary.png",
        top_n=15,
    )

    # 9b. Per-state bar charts
    print(f"  Generating state bar charts ({len(state_profiles)} states)...")
    for state_code, profile in state_profiles.items():
        plot_state_bar(
            state_code,
            profile,
            output_path=output_dir / f"shap_state_bar_{state_code}.png",
            top_n=10,
        )
    print(f"  Saved {len(state_profiles)} state bar charts")

    # 9c. Alignment heatmap (only if EPS data available)
    if eps_df is not None and alignment_results:
        plot_alignment_heatmap(
            alignment_results, eps_df,
            output_path=output_dir / "shap_alignment_heatmap.png",
        )

    # ── 10. Console summary table ──
    print_section_header("SHAP-EPS ALIGNMENT SUMMARY")

    if eps_df is not None and alignment_results:
        header = f"{'State':<7} {'EPS_Rank':>8}  {'Dominant_SHAP_Feature':<25} {'Alignment':>9}  {'Verdict':<8}"
        print(header)
        print("-" * len(header))

        eps_sorted = eps_df.sort_values('EPS_rank')
        for _, row in eps_sorted.iterrows():
            state = str(row['customer_state'])
            eps_rank = int(row['EPS_rank'])
            dominant = state_profiles.get(state, {}).get('dominant_feature', '?')
            ar = alignment_results.get(state, {})
            score = ar.get('overall_score', 0.0)
            verdict = ar.get('verdict', '?')

            # Flag data_sparse states
            data_sparse = bool(row.get('data_sparse', False))
            flag = " ← data_sparse" if data_sparse and verdict == "LOW" else ""

            print(
                f"{state:<7} {eps_rank:>8}  {dominant:<25} {score:>9.2f}  {verdict:<8}{flag}"
            )
    else:
        print("  [Skipped] EPS data not available — alignment summary not generated.")

    print_section_header("SHAP EXPLAINER PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
