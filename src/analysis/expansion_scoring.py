#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expansion Priority Score (EPS) Calculation Pipeline

This module calculates the Expansion Priority Score (EPS) for Olist Brazilian states
using a combination of five opportunity components (Demand, Growth, Penetration, Momentum, Logistics)
and optimizes their weights using Sequential Least Squares Programming (SLSQP) to maximize Shannon entropy.

Mathematical formulation:
- OPP = Sum(w_c * Component_c)
- Risk Adjustment = 1 - gamma * Logistics_Cost_normalized
- EPS = OPP * Risk Adjustment (scaled to 0 - 100)

Usage:
    python src/analysis/expansion_scoring.py [options]
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import spearmanr

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# Configure utf-8 encoding for stdout on Windows console to avoid UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to PYTHONPATH for standalone execution to avoid import errors
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.system_utils import print_section_header


def load_and_prepare_data(
    features_path: Path,
    pred_path: Path,
    n_recent_weeks: int
) -> pd.DataFrame:
    """
    Loads weekly features and next-week predictions, filters for the most recent
    weeks, aggregates data per state, and merges predictions.

    Args:
        features_path (Path): Path to the weekly features CSV file.
        pred_path (Path): Path to the predicted next week revenue CSV file.
        n_recent_weeks (int): Number of recent weeks to use for state-level averages.

    Returns:
        pd.DataFrame: Aggregated metrics by customer state.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at: {features_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_path}")

    print(f"-> Loading weekly features: {features_path.name}")
    features = pd.read_csv(features_path)
    print(f"-> Loading revenue predictions: {pred_path.name}")
    pred = pd.read_csv(pred_path)

    # Parse dates
    features["week_start"] = pd.to_datetime(
        features["year_week"].astype(str).str.split("/").str[0],
        errors="coerce"
    )

    # Determine latest N weeks
    latest_weeks = (
        features["week_start"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tail(n_recent_weeks)
    )
    weeks_list = latest_weeks.dt.strftime('%Y-%m-%d').tolist()
    print(f"-> Weeks utilized for recent state metrics: {weeks_list}")

    # Filter to recent weeks
    df_recent = features[features["week_start"].isin(latest_weeks)].copy()

    # Aggregate by customer state
    df_state = df_recent.groupby("customer_state").agg(
        rev_mean            = ("revenue",            "mean"),
        seller_mean         = ("unique_sellers",      "mean"),
        freight_mean        = ("avg_freight_value",   "mean"),
        population          = ("population",          "last"),
        gdp_per_capita      = ("gdp_per_capita",      "last"),
        revenue_rolling_8   = ("revenue_rolling_8",   "last"),
    ).reset_index()

    # Normalize prediction column name
    if "predicted_revenue" in pred.columns:
        pred = pred.rename(columns={"predicted_revenue": "predicted_next_week_revenue"})

    # Merge predictions
    df_state = df_state.merge(
        pred[["customer_state", "predicted_next_week_revenue"]],
        on="customer_state", how="left"
    )

    print(f"-> Successfully prepared and aggregated data for {len(df_state)} states.")
    return df_state


def softclip_positive(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Smooth approximation of max(x, 0): ln(1 + exp(k * x)) / k.
    Clips input value inside exp to avoid overflow.
    """
    return np.log1p(np.exp(np.clip(k * x, -50, 50))) / k


def calculate_raw_components(
    df: pd.DataFrame,
    min_sellers: int
) -> pd.DataFrame:
    """
    Computes raw expansion score components from aggregated state data.

    Calculations:
    - 2A. PD (Predicted Demand): 0.5 * ln(1 + predicted_rev) + 0.5 * ln(1 + R4w)
    - 2B. GP (Growth Potential): (R4w - R8w) / R8w clipped to [-1, 1]
    - 2C. PG (Penetration Gap): GDP-weighted expected revenue comparison, softclipped
    - 2D. MMI (Market Momentum Index): ln(1 + revenue/seller) if unique_sellers >= min_sellers
    - 2E. LC (Logistics Cost): raw freight cost

    Args:
        df (pd.DataFrame): Aggregated state-level data.
        min_sellers (int): Minimum sellers to validate MMI component.

    Returns:
        pd.DataFrame: DataFrame containing raw component values per state.
    """
    print("-> Computing raw expansion score components...")
    
    df = df.copy()
    df['R4w']                = df['rev_mean']
    df['R8w']                = df['revenue_rolling_8']
    df['predicted_rev']      = df['predicted_next_week_revenue'].fillna(0)
    df['pop']                = df['population']
    df['unique_sellers_R4w'] = df['seller_mean']
    df['avg_freight']        = df['freight_mean']

    out = pd.DataFrame(index=df.index)
    out['customer_state'] = df['customer_state']

    # 2A. PD (Predicted Demand)
    out['PD_raw'] = (
        0.5 * np.log1p(df['predicted_rev']) +
        0.5 * np.log1p(df['R4w'])
    )

    # 2B. GP (Growth Potential)
    nat_median_r8w = df['R8w'].median()
    epsilon        = nat_median_r8w * 0.01 if pd.notnull(nat_median_r8w) else 1.0
    R8w_safe       = df['R8w'].clip(lower=epsilon)
    out['GP_raw']  = ((df['R4w'] - R8w_safe) / R8w_safe).clip(-1, 1).fillna(0)
    out['data_sparse'] = df['R8w'] < df['R8w'].quantile(0.05)

    # 2C. PG (Penetration Gap)
    nat_gdp_pc   = np.average(df['gdp_per_capita'], weights=df['pop'])
    gdp_weight   = df['gdp_per_capita'] / nat_gdp_pc
    nat_rev_pc   = df['R4w'].sum() / df['pop'].sum()
    expected_rev = df['pop'] * nat_rev_pc * gdp_weight

    pg_raw = (expected_rev - df['R4w']) / (expected_rev + 1e-9)
    out['PG_raw'] = pg_raw
    out['PG_raw_soft'] = softclip_positive(pg_raw.values)

    # 2D. MMI (Market Momentum Index)
    seller_ok      = df['unique_sellers_R4w'] >= min_sellers
    rev_per_seller = df['R4w'] / df['unique_sellers_R4w'].replace(0, np.nan)
    out['MMI_raw'] = np.where(seller_ok, np.log1p(rev_per_seller), np.nan)

    # 2E. LC (Logistics Cost)
    out['LC_raw'] = df['avg_freight'].values

    return out


def normalize_zscore_to_01(series: pd.Series) -> pd.Series:
    """
    Two-step normalisation:
      1. Z-score: reduces outlier influence.
      2. Min-max to [0,1]: unified scale across components.
    NaN values are ignored during mean/std calculation, but preserved in output.
    """
    valid = series.dropna()
    if valid.empty:
        return series.fillna(0.0)
    mu, sigma = valid.mean(), valid.std()
    z = (series - mu) / (sigma + 1e-9)
    z_min, z_max = z.min(), z.max()
    return (z - z_min) / (z_max - z_min + 1e-9)


def normalize_components(
    out_df: pd.DataFrame,
    comp_opp: List[str]
) -> pd.DataFrame:
    """
    Normalizes all computed raw components using the Z-score to [0, 1] pipeline.
    Imputes missing MMI values with the median.
    """
    print("-> Normalizing components to [0, 1] scale...")
    norm = pd.DataFrame(index=out_df.index)
    norm['customer_state'] = out_df['customer_state']

    norm['PD'] = normalize_zscore_to_01(out_df['PD_raw'])
    norm['GP'] = normalize_zscore_to_01(out_df['GP_raw'])
    norm['PG'] = normalize_zscore_to_01(out_df['PG_raw_soft'])

    # MMI z-score normalization and median imputation
    norm['MMI'] = normalize_zscore_to_01(out_df['MMI_raw'])
    norm['MMI'] = norm['MMI'].fillna(norm['MMI'].median())

    # LC (Logistics Cost) normalization
    norm['LC'] = normalize_zscore_to_01(out_df['LC_raw'])

    # Validate that all values are bounded between 0 and 1
    for col in comp_opp + ['LC']:
        norm[col] = norm[col].clip(0.0, 1.0)

    return norm


def arithmetic_opp(weights: np.ndarray, comp_matrix: np.ndarray) -> np.ndarray:
    """
    Computes pre-risk opportunity score as the weighted sum of component values.
    """
    return (comp_matrix * weights).sum(axis=1)


def find_optimal_weights(
    norm_df: pd.DataFrame,
    comp_opp: List[str],
    constraints: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Maximise normalised Shannon entropy of the OPP distribution:
        max  H = −(1/ln n) · Σ p(s) · ln p(s)
        s.t. lo_c ≤ w_c ≤ hi_c  ∀c
             Σ w_c = 1
    Algorithm: SLSQP (Sequential Least Squares Programming).
    Initialisation: midpoint of each constraint interval, normalised to sum=1.
    """
    print("-> Optimizing component weights using SLSQP (Entropy Maximization)...")
    bounds = [constraints[c] for c in comp_opp]
    comp_matrix = norm_df[comp_opp].values

    def neg_entropy(w: np.ndarray) -> float:
        scores = arithmetic_opp(w, comp_matrix)
        total  = scores.sum()
        if total <= 0:
            return 0.0
        p = scores / total
        n = len(p)
        return (1.0 / np.log(n)) * np.sum(p * np.log(p + 1e-10))

    w0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
    w0 /= w0.sum()

    result = minimize(
        neg_entropy, w0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
        options={'ftol': 1e-9, 'maxiter': 1000},
    )
    if not result.success:
        raise ValueError(f"Optimisation failed: {result.message}")

    return result.x


def compute_eps_scores(
    norm_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    w_star: np.ndarray,
    gamma: float,
    comp_opp: List[str]
) -> pd.DataFrame:
    """
    Calculates final EPS scores and ranks states.
    - OPP score (weighted sum of components)
    - Risk adjustment: 1 - gamma * LC_norm
    - Final EPS score: OPP * Risk_Adj, normalized to [0, 100]
    """
    print("-> Calculating final EPS scores and rankings...")
    opp_scores = arithmetic_opp(w_star, norm_df[comp_opp].values)
    risk_adj   = 1.0 - gamma * norm_df['LC'].values
    eps_raw    = opp_scores * risk_adj

    # Rescale to [0, 100]
    eps_min, eps_max = eps_raw.min(), eps_raw.max()
    eps_score = (eps_raw - eps_min) / (eps_max - eps_min + 1e-9) * 100.0

    result = raw_df[['customer_state', 'data_sparse']].copy()
    result['EPS_score'] = eps_score
    result['EPS_rank']  = pd.Series(eps_score).rank(ascending=False).astype(int).values
    result['OPP_score'] = opp_scores
    result['Risk_Adj']  = risk_adj

    for comp in comp_opp:
        result[f'{comp}_norm'] = norm_df[comp].values
    result['LC_norm'] = norm_df['LC'].values

    for i, c in enumerate(comp_opp):
        result[f'w_{c}'] = round(w_star[i], 4)

    return result.sort_values('EPS_rank').reset_index(drop=True)


def save_outputs(
    w_star: np.ndarray,
    result_df: pd.DataFrame,
    output_dir: Path,
    comp_opp: List[str],
    gamma: float
) -> None:
    """
    Saves weight config JSON and scoring results CSV to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    w_star_path = output_dir / "w_star.json"
    eps_path    = output_dir / "eps_results.csv"

    w_dict = {c: float(w) for c, w in zip(comp_opp, w_star)}
    with open(w_star_path, 'w') as f:
        json.dump({"w_star": w_dict, "gamma": gamma}, f, indent=2)
    print(f"-> Saved optimal weights to: {w_star_path}")

    result_df.to_csv(eps_path, index=False)
    print(f"-> Saved EPS scoring results to: {eps_path}")


# ── Plotting / Visualizations ───────────────────────────────────────────────

def plot_component_contributions(
    result_df: pd.DataFrame,
    w_star: np.ndarray,
    comp_opp: List[str],
    figure_dir: Path
) -> None:
    """
    Plots the weighted components bar chart for each state and risk adjustment factor overlay.
    """
    print("-> Generating component contribution bar chart...")
    fig, ax = plt.subplots(figsize=(14, 6))

    states  = result_df['customer_state'].values
    x       = np.arange(len(states))
    width   = 0.18
    colors  = {'PD': '#4C72B0', 'GP': '#DD8452', 'PG': '#55A868', 'MMI': '#C44E52'}
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for comp, offset in zip(comp_opp, offsets):
        vals = result_df[f'{comp}_norm'].values * w_star[comp_opp.index(comp)]
        ax.bar(x + offset * width, vals, width, label=f"{comp} (w={w_star[comp_opp.index(comp)]:.2f})",
               color=colors[comp], alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, result_df['Risk_Adj'].values, color='black', linewidth=1.5,
             linestyle='--', label='Risk Adjustment (1−γ·LC)', marker='o', markersize=3)
    ax2.set_ylim(0.7, 1.05)
    ax2.set_ylabel('Risk Adjustment factor', fontsize=10)
    ax2.legend(loc='lower right', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weighted component score', fontsize=10)
    ax.set_title('EPS: Component contributions by state (ranked by EPS score)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = figure_dir / "fig1_component_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 1 to: {fig_path}")


def plot_choropleth(
    result_df: pd.DataFrame,
    geo_path: Path,
    figure_dir: Path
) -> None:
    """
    Loads geojson boundaries, merges EPS results, and plots three choropleth maps:
    1. EPS Score, 2. Opportunity Score (OPP), 3. Logistics Cost (LC).
    """
    if gpd is None:
        print("Warning: geopandas is not installed. Skipping choropleth plotting.")
        return

    print("-> Generating spatial distribution choropleth maps...")
    
    # Load geometry
    try:
        import geobr
        print("   Using geobr library to download state boundaries...")
        gdf_states = geobr.read_state(year=2020)
        state_key  = 'abbrev_state'
    except Exception as e:
        print(f"   geobr not available ({e}). Falling back to local GeoJSON file...")
        if not geo_path.exists():
            print(f"   Warning: GeoJSON file not found at {geo_path}. Skipping choropleth plotting.")
            return
        gdf_states = gpd.read_file(geo_path)
        # Detect key column
        candidates = [c for c in gdf_states.columns if gdf_states[c].astype(str).str.len().max() == 2]
        state_key  = candidates[0] if candidates else 'sigla'
        print(f"   Using geometry key column: '{state_key}' from {geo_path.name}")

    # Merge EPS results
    gdf_states[state_key] = gdf_states[state_key].astype(str).str.upper().str.strip()
    
    res_copy = result_df.copy()
    res_copy['customer_state'] = res_copy['customer_state'].astype(str).str.upper().str.strip()

    gdf = gdf_states.merge(res_copy, left_on=state_key, right_on='customer_state', how='left')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    plot_configs = [
        ('EPS_score',  'EPS Score (0–100)',        'YlOrRd'),
        ('OPP_score',  'Opportunity Score (OPP)',  'Blues'),
        ('LC_norm',    'Logistics Cost (normalised)','Reds'),
    ]

    for ax, (col, title, cmap) in zip(axes, plot_configs):
        gdf.plot(
            column=col, ax=ax, cmap=cmap,
            legend=True,
            legend_kwds={'label': title, 'orientation': 'horizontal', 'shrink': 0.7},
            missing_kwds={'color': 'lightgrey', 'label': 'No data'},
            edgecolor='white', linewidth=0.4
        )
        # Annotate state abbreviations
        for _, row in gdf.iterrows():
            if pd.notnull(row.get(col)) and row.geometry is not None:
                try:
                    centroid = row.geometry.centroid
                    ax.annotate(
                        text=row[state_key],
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color='#333333'
                    )
                except Exception:
                    pass
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis('off')

    plt.suptitle('EPS: Spatial distribution across Brazilian states', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = figure_dir / "fig2_choropleth.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 2 to: {fig_path}")


def plot_radar_profiles(
    result_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    comp_opp: List[str],
    figure_dir: Path,
    top_n: int = 6
) -> None:
    """
    Generates radar profiles for the Top-N states.
    """
    print(f"-> Generating radar profiles for Top-{top_n} states...")
    top_states = result_df.sort_values('EPS_rank').head(top_n)
    labels     = ['PD', 'GP', 'PG', 'MMI', 'Logistics\nQuality']
    n_vars     = len(labels)
    angles     = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles    += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
    fig.suptitle('EPS: Component radar profiles — Top 6 states', fontsize=13)
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_states.iterrows()):
        ax = axes[i]
        state = row['customer_state']
        vals  = [
            norm_df.loc[norm_df['customer_state'] == state, comp].values[0]
            for comp in comp_opp
        ] + [1.0 - row['LC_norm']]
        vals += vals[:1]

        ax.plot(angles, vals, color='steelblue', linewidth=2)
        ax.fill(angles, vals, color='steelblue', alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=6, color='grey')
        ax.set_title(
            f"Rank #{int(row['EPS_rank'])}: {state}\nEPS={row['EPS_score']:.1f}",
            size=10, weight='bold', pad=14
        )
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = figure_dir / "fig3_radar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 3 to: {fig_path}")


def plot_correlation_heatmap(
    result_df: pd.DataFrame,
    figure_dir: Path
) -> None:
    """
    Generates correlation heatmap of the normalized components and final scores.
    """
    print("-> Generating correlation heatmap...")
    corr_cols = ['PD_norm', 'GP_norm', 'PG_norm', 'MMI_norm', 'LC_norm', 'OPP_score', 'EPS_score']
    existing_cols = [col for col in corr_cols if col in result_df.columns]
    if len(existing_cols) < 2:
        print("Warning: Not enough correlation columns exist. Skipping correlation heatmap.")
        return

    corr_matrix = result_df[existing_cols].corr(method='pearson')

    rename_dict = {
        'PD_norm': 'PD (Demand)',
        'GP_norm': 'GP (Growth)',
        'PG_norm': 'PG (Penetration)',
        'MMI_norm': 'MMI (Momentum)',
        'LC_norm': 'LC (Logistics Cost)',
        'OPP_score': 'OPP (Pre-risk)',
        'EPS_score': 'EPS (Final)'
    }
    
    active_rename = {k: v for k, v in rename_dict.items() if k in existing_cols}
    corr_matrix = corr_matrix.rename(columns=active_rename, index=active_rename)

    sns.set_theme(style="white")
    fig = plt.figure(figsize=(9, 7))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"}
    )

    plt.title('Correlation Heatmap of EPS Components & Final Scores', fontsize=12, pad=15)
    plt.tight_layout()
    fig_path = figure_dir / "fig3b_correlation_heatmap.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 3b to: {fig_path}")


# ── Sensitivity Analysis ─────────────────────────────────────────────────────

def run_monte_carlo(
    norm_df: pd.DataFrame,
    w_star: np.ndarray,
    constraints: Dict[str, Tuple[float, float]],
    comp_opp: List[str],
    rank_base: np.ndarray,
    gamma: float,
    figure_dir: Path,
    n_sim: int = 10000,
    seed: int = 42
) -> None:
    """
    Runs Monte Carlo simulation perturbing weights to check rank stability.
    """
    print(f"-> Running Monte Carlo stability simulation (n={n_sim:,} iterations)...")
    rng = np.random.default_rng(seed)
    bounds_lo = np.array([constraints[c][0] for c in comp_opp])
    bounds_hi = np.array([constraints[c][1] for c in comp_opp])
    rho_list = []

    for _ in range(n_sim):
        w_sim  = rng.dirichlet(w_star * 50)
        w_sim  = np.clip(w_sim, bounds_lo, bounds_hi)
        w_sim /= w_sim.sum()

        eps_sim  = compute_eps_from_weights(w_sim, norm_df, comp_opp, gamma)
        rank_sim = pd.Series(eps_sim).rank(ascending=False).values
        rho, _   = spearmanr(rank_base, rank_sim)
        rho_list.append(rho)

    rho_array = np.array(rho_list)
    mean_rho  = rho_array.mean()
    pct_above_095 = (rho_array > 0.95).mean() * 100
    pct_above_090 = (rho_array > 0.90).mean() * 100
    verdict = "ROBUST" if pct_above_095 >= 95 else ("MODERATE" if pct_above_090 >= 80 else "SENSITIVE")

    print("\n   Monte Carlo result:")
    print(f"     Mean ρ         : {mean_rho:.4f}")
    print(f"     % sims ρ>0.95  : {pct_above_095:.1f}%")
    print(f"     % sims ρ>0.90  : {pct_above_090:.1f}%")
    print(f"     Verdict        : {verdict}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rho_array, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0.95, color='darkred',  linestyle='--', label='ρ=0.95 (ROBUST threshold)')
    ax.axvline(0.90, color='orange',   linestyle='--', label='ρ=0.90 (MODERATE threshold)')
    ax.axvline(mean_rho, color='green', linestyle='-', linewidth=1.8,
               label=f'Mean ρ={mean_rho:.4f}')
    ax.set_xlabel('Spearman ρ (rank correlation vs baseline)')
    ax.set_ylabel('Count')
    ax.set_title(f'Monte Carlo sensitivity (n={n_sim:,} simulations)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = figure_dir / "fig4_monte_carlo.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 4 to: {fig_path}")


def run_oat_sweep(
    norm_df: pd.DataFrame,
    w_star: np.ndarray,
    constraints: Dict[str, Tuple[float, float]],
    comp_opp: List[str],
    rank_base: np.ndarray,
    gamma: float,
    figure_dir: Path,
    n_steps: int = 40
) -> None:
    """
    Runs One-At-a-Time (OAT) parameter sweep.
    """
    print("-> Running One-At-a-Time (OAT) sensitivity sweep...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('OAT Weight Sweep — Spearman ρ vs weight value', fontsize=12)
    axes  = axes.flatten()
    stats = {}

    for i, comp in enumerate(comp_opp):
        lo, hi = constraints[comp]
        sweep  = np.linspace(lo, hi, n_steps)
        rho_list = []

        for w_val in sweep:
            w_new    = w_star.copy()
            w_new[i] = w_val
            others   = [j for j in range(len(comp_opp)) if j != i]
            s_others = w_new[others].sum()
            if s_others > 0:
                w_new[others] *= (1.0 - w_val) / s_others

            eps_sim  = compute_eps_from_weights(w_new, norm_df, comp_opp, gamma)
            rho, _   = spearmanr(rank_base, pd.Series(eps_sim).rank(ascending=False).values)
            rho_list.append(rho)

        min_rho = min(rho_list)
        stats[comp] = {'min_rho': min_rho, 'range': (lo, hi)}

        ax = axes[i]
        ax.plot(sweep, rho_list, marker='o', markersize=3, linestyle='-', color='teal')
        ax.axhline(0.90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='ρ=0.90')
        ax.axvline(w_star[i], color='green', linestyle=':', linewidth=1.2, label=f'w*={w_star[i]:.3f}')
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(0.80, min_rho - 0.02), 1.02)
        ax.set_title(f'w({comp})  range=[{lo},{hi}]', fontsize=10)
        ax.set_xlabel(f'w_{comp}')
        ax.set_ylabel('Spearman ρ')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = figure_dir / "fig5_oat_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 5 to: {fig_path}")

    print("\n   OAT minimum ρ per component:")
    for comp, s in stats.items():
        flag = "✓ SAFE" if s['min_rho'] >= 0.85 else "⚠ SENSITIVE"
        print(f"     {comp:4s}: min ρ = {s['min_rho']:.4f}  {flag}")


def run_gamma_sweep(
    norm_df: pd.DataFrame,
    w_star: np.ndarray,
    comp_opp: List[str],
    rank_base: np.ndarray,
    current_gamma: float,
    result_df: pd.DataFrame,
    figure_dir: Path
) -> None:
    """
    Runs sensitivity sweep for Gamma (risk weight parameter).
    """
    print("-> Running Gamma parameter sweep...")
    gamma_range = np.linspace(0.05, 0.40, 30)
    rho_gamma   = []
    rank_shifts = []

    for g in gamma_range:
        eps_g    = compute_eps_from_weights(w_star, norm_df, comp_opp, gamma=g)
        rank_g   = pd.Series(eps_g).rank(ascending=False).values
        rho, _   = spearmanr(rank_base, rank_g)
        rho_gamma.append(rho)
        n_shifted = (np.abs(rank_g - rank_base) >= 3).sum()
        rank_shifts.append(n_shifted)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Sensitivity C: Gamma (risk weight) sweep', fontsize=12)

    ax1.plot(gamma_range, rho_gamma, color='steelblue', linewidth=2, marker='o', markersize=3)
    ax1.axvline(current_gamma, color='green', linestyle='--', linewidth=1.5, label=f'γ={current_gamma} (current)')
    ax1.axhline(0.90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='ρ=0.90')
    ax1.set_xlabel('γ (risk penalty weight)')
    ax1.set_ylabel('Spearman ρ vs baseline')
    ax1.set_title('Rank stability across γ values')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(gamma_range, rank_shifts, color='coral', linewidth=2, marker='s', markersize=3)
    ax2.axvline(current_gamma, color='green', linestyle='--', linewidth=1.5, label=f'γ={current_gamma} (current)')
    ax2.set_xlabel('γ (risk penalty weight)')
    ax2.set_ylabel('# states with |Δrank| ≥ 3')
    ax2.set_title('States shifting ≥3 positions')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = figure_dir / "fig6_gamma_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved Fig 6 to: {fig_path}")

    # Analyze top states most sensitive
    eps_low   = compute_eps_from_weights(w_star, norm_df, comp_opp, gamma=0.05)
    eps_high  = compute_eps_from_weights(w_star, norm_df, comp_opp, gamma=0.40)
    rank_low  = pd.Series(eps_low).rank(ascending=False).values
    rank_high = pd.Series(eps_high).rank(ascending=False).values
    delta     = np.abs(rank_high - rank_low)

    gamma_sensitivity = pd.DataFrame({
        'state':      result_df['customer_state'].values,
        'rank_γ005':  rank_low.astype(int),
        'rank_γ040':  rank_high.astype(int),
        'delta_rank': delta.astype(int),
        'LC_norm':    norm_df['LC'].values,
    }).sort_values('delta_rank', ascending=False)

    print("\n   States most sensitive to γ changes (γ=0.05 vs γ=0.40):")
    print(gamma_sensitivity.head(10).to_string(index=False))


def compute_eps_from_weights(
    w: np.ndarray,
    norm_df: pd.DataFrame,
    comp_opp: List[str],
    gamma: float
) -> np.ndarray:
    """
    Utility function to compute EPS scores directly from raw weight array and norm dataframe.
    Used for sensitivity sweeps.
    """
    opp     = arithmetic_opp(w, norm_df[comp_opp].values)
    eps_raw = opp * (1.0 - gamma * norm_df['LC'].values)
    eps_min, eps_max = eps_raw.min(), eps_raw.max()
    return (eps_raw - eps_min) / (eps_max - eps_min + 1e-9) * 100.0


# ── Main Pipeline Execution ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Expansion Priority Score (EPS) for Olist Brazilian states."
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(project_root / "data" / "processed" / "olist" / "features_weekly.csv"),
        help="Path to features_weekly.csv"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(project_root / "data" / "processed" / "olist" / "predicted_next_week_revenue.csv"),
        help="Path to predicted_next_week_revenue.csv"
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default=str(project_root / "data" / "external" / "br_states.geojson"),
        help="Path to brazil_states.geojson or br_states.geojson"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "eps"),
        help="Directory to save EPS results and weights JSON"
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(project_root / "reports" / "figures"),
        help="Directory to save generated plots"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.20,
        help="Risk parameter gamma (default: 0.20)"
    )
    parser.add_argument(
        "--min-sellers",
        type=int,
        default=5,
        help="Minimum sellers to validate MMI (default: 5)"
    )
    parser.add_argument(
        "--n-weeks",
        type=int,
        default=4,
        help="Number of recent weeks to average metrics over (default: 4)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, skips generating and saving plots"
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="If set, skips running sensitivity analyses"
    )

    args = parser.parse_args()

    features_path = Path(args.features)
    pred_path     = Path(args.predictions)
    geo_path      = Path(args.geojson)
    output_dir    = Path(args.output_dir)
    figure_dir    = Path(args.figure_dir)

    print_section_header("RUNNING EXPANSION PRIORITY SCORING PIPELINE")
    print(f"Hyperparameters: GAMMA={args.gamma}, MIN_SELLERS={args.min_sellers}, N_RECENT_WEEKS={args.n_weeks}")

    # Define optimization variables
    comp_opp = ['PD', 'GP', 'PG', 'MMI']
    constraints = {
        'PD':  (0.25, 0.45),
        'GP':  (0.15, 0.35),
        'PG':  (0.15, 0.30),
        'MMI': (0.05, 0.15),
    }

    # 1. Load and aggregate features
    df_state = load_and_prepare_data(features_path, pred_path, args.n_weeks)

    # 2. Calculate raw components
    raw_df = calculate_raw_components(df_state, args.min_sellers)

    # 3. Normalize raw components to [0, 1]
    norm_df = normalize_components(raw_df, comp_opp)

    # 4. Optimize component weights
    w_star = find_optimal_weights(norm_df, comp_opp, constraints)

    print_section_header("OPTIMAL COMPONENT WEIGHTS (w*)")
    for c, w in zip(comp_opp, w_star):
        lo, hi = constraints[c]
        bar = "█" * int(w * 40)
        print(f"  w({c:3s}) = {w:.4f}  [{lo:.2f},{hi:.2f}]  {bar}")

    # 5. Calculate EPS scores & rankings
    result_df = compute_eps_scores(norm_df, raw_df, w_star, args.gamma, comp_opp)

    # 6. Save results
    save_outputs(w_star, result_df, output_dir, comp_opp, args.gamma)

    # 7. Print summary of Top 10 States
    print_section_header("EPS TOP 10 STATES SUMMARY")
    cols_to_print = ['EPS_rank', 'customer_state', 'EPS_score', 'OPP_score', 'Risk_Adj']
    print(result_df[cols_to_print].head(10).to_string(index=False))

    # 8. Plotting (if enabled)
    if not args.no_plots:
        print_section_header("GENERATING VISUALIZATIONS")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_component_contributions(result_df, w_star, comp_opp, figure_dir)
        plot_choropleth(result_df, geo_path, figure_dir)
        plot_radar_profiles(result_df, norm_df, comp_opp, figure_dir, top_n=6)
        plot_correlation_heatmap(result_df, figure_dir)

    # 9. Sensitivity analysis (if enabled)
    if not args.no_sensitivity:
        print_section_header("RUNNING SENSITIVITY ANALYSES")
        # Ensure aligned state indexes for baseline rank comparisons
        norm_sorted = norm_df.sort_values('customer_state').reset_index(drop=True)
        result_sorted = result_df.sort_values('customer_state').reset_index(drop=True)
        rank_base = result_sorted['EPS_rank'].values
        
        # Monte Carlo Simulation
        run_monte_carlo(norm_sorted, w_star, constraints, comp_opp, rank_base, args.gamma, figure_dir)
        
        # OAT Sweep
        run_oat_sweep(norm_sorted, w_star, constraints, comp_opp, rank_base, args.gamma, figure_dir)
        
        # Gamma Sweep
        run_gamma_sweep(norm_sorted, w_star, comp_opp, rank_base, args.gamma, result_sorted, figure_dir)

    print_section_header("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
