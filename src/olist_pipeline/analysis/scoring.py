#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expansion Priority Score (EPS) Calculation Pipeline
Calculates state-level priority scores using entropy-optimized weights.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import spearmanr

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from src.olist_pipeline.utils.system_utils import print_section_header
from src.olist_pipeline.utils.logger import setup_logger
from src.olist_pipeline.utils.math_utils import softclip_positive, normalize_zscore_to_01

logger = setup_logger("expansion_scoring")

def load_and_prepare_data(features_path: Path, pred_path: Path, n_recent_weeks: int) -> pd.DataFrame:
    """
    Loads weekly features and next-week predictions, aggregating metrics per state.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    logger.info(f"Loading features from {features_path.name} and predictions from {pred_path.name}...")
    features = pd.read_csv(features_path)
    pred = pd.read_csv(pred_path)

    features["week_start"] = pd.to_datetime(features["year_week"].astype(str).str.split("/").str[0], errors="coerce")
    latest_weeks = features["week_start"].dropna().drop_duplicates().sort_values().tail(n_recent_weeks)
    
    df_recent = features[features["week_start"].isin(latest_weeks)].copy()

    df_state = df_recent.groupby("customer_state").agg(
        rev_mean            = ("revenue",            "mean"),
        seller_mean         = ("unique_sellers",      "mean"),
        freight_mean        = ("avg_freight_value",   "mean"),
        population          = ("population",          "last"),
        gdp_per_capita      = ("gdp_per_capita",      "last"),
        revenue_rolling_8   = ("revenue_rolling_8",   "last"),
    ).reset_index()

    if "predicted_revenue" in pred.columns:
        pred = pred.rename(columns={"predicted_revenue": "predicted_next_week_revenue"})

    df_state = df_state.merge(pred[["customer_state", "predicted_next_week_revenue"]], on="customer_state", how="left")
    logger.info(f"Prepared data for {len(df_state)} states.")
    return df_state


def calculate_raw_components(df: pd.DataFrame, min_sellers: int) -> pd.DataFrame:
    """
    Computes raw expansion score components (PD, GP, PG, MMI, LC).
    """
    logger.info("Computing raw components...")
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    out['customer_state'] = df['customer_state']

    # 2A. PD (Predicted Demand)
    out['PD_raw'] = 0.5 * np.log1p(df['predicted_next_week_revenue'].fillna(0)) + 0.5 * np.log1p(df['rev_mean'])

    # 2B. GP (Growth Potential)
    nat_median_r8w = df['revenue_rolling_8'].median()
    epsilon = nat_median_r8w * 0.01 if pd.notnull(nat_median_r8w) else 1.0
    R8w_safe = df['revenue_rolling_8'].clip(lower=epsilon)
    out['GP_raw']  = ((df['rev_mean'] - R8w_safe) / R8w_safe).clip(-1, 1).fillna(0)
    out['data_sparse'] = df['revenue_rolling_8'] < df['revenue_rolling_8'].quantile(0.05)

    # 2C. PG (Penetration Gap)
    nat_gdp_pc   = np.average(df['gdp_per_capita'], weights=df['population'])
    gdp_weight   = df['gdp_per_capita'] / nat_gdp_pc
    nat_rev_pc   = df['rev_mean'].sum() / df['population'].sum()
    expected_rev = df['population'] * nat_rev_pc * gdp_weight
    pg_raw = (expected_rev - df['rev_mean']) / (expected_rev + 1e-9)
    out['PG_raw'] = pg_raw
    out['PG_raw_soft'] = softclip_positive(pg_raw.values)

    # 2D. MMI (Market Momentum Index)
    seller_ok      = df['seller_mean'] >= min_sellers
    rev_per_seller = df['rev_mean'] / df['seller_mean'].replace(0, np.nan)
    out['MMI_raw'] = np.where(seller_ok, np.log1p(rev_per_seller), np.nan)

    # 2E. LC (Logistics Cost)
    out['LC_raw'] = df['freight_mean'].values

    return out


def normalize_components(out_df: pd.DataFrame, comp_opp: List[str]) -> pd.DataFrame:
    """
    Normalizes raw components to a unified [0, 1] scale.
    """
    logger.info("Normalizing components...")
    norm = pd.DataFrame(index=out_df.index)
    norm['customer_state'] = out_df['customer_state']

    norm['PD'] = normalize_zscore_to_01(out_df['PD_raw'])
    norm['GP'] = normalize_zscore_to_01(out_df['GP_raw'])
    norm['PG'] = normalize_zscore_to_01(out_df['PG_raw_soft'])
    norm['MMI'] = normalize_zscore_to_01(out_df['MMI_raw']).fillna(0.5) # Median fallback
    norm['LC'] = normalize_zscore_to_01(out_df['LC_raw'])

    for col in comp_opp + ['LC']:
        norm[col] = norm[col].clip(0.0, 1.0)

    return norm


def find_optimal_weights(norm_df: pd.DataFrame, comp_opp: List[str], constraints: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Optimizes weights using Entropy Maximization (SLSQP).
    """
    logger.info("Optimizing weights via SLSQP...")
    bounds = [constraints[c] for c in comp_opp]
    comp_matrix = norm_df[comp_opp].values

    def neg_entropy(w: np.ndarray) -> float:
        scores = (comp_matrix * w).sum(axis=1)
        total = scores.sum()
        if total <= 0: return 0.0
        p = scores / total
        return (1.0 / np.log(len(p))) * np.sum(p * np.log(p + 1e-10))

    w0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
    w0 /= w0.sum()

    result = minimize(neg_entropy, w0, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0})
    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")
    
    return result.x


def compute_eps_scores(norm_df: pd.DataFrame, raw_df: pd.DataFrame, w_star: np.ndarray, gamma: float, comp_opp: List[str]) -> pd.DataFrame:
    """
    Calculates final EPS scores and generates rankings.
    """
    logger.info("Calculating final EPS rankings...")
    opp_scores = (norm_df[comp_opp].values * w_star).sum(axis=1)
    risk_adj   = 1.0 - gamma * norm_df['LC'].values
    eps_raw    = opp_scores * risk_adj

    eps_min, eps_max = eps_raw.min(), eps_raw.max()
    eps_score = (eps_raw - eps_min) / (eps_max - eps_min + 1e-9) * 100.0

    result = raw_df[['customer_state', 'data_sparse']].copy()
    result['EPS_score'] = eps_score
    result['EPS_rank']  = pd.Series(eps_score).rank(ascending=False).astype(int).values
    result['OPP_score'] = opp_scores
    result['Risk_Adj']  = risk_adj

    for i, comp in enumerate(comp_opp):
        result[f'{comp}_norm'] = norm_df[comp].values
        result[f'w_{comp}'] = round(w_star[i], 4)
    result['LC_norm'] = norm_df['LC'].values

    return result.sort_values('EPS_rank').reset_index(drop=True)


def plot_results(result_df: pd.DataFrame, w_star: np.ndarray, comp_opp: List[str], figure_dir: Path) -> None:
    """
    Generates component contribution visualizations.
    """
    logger.info("Generating visualizations...")
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Component Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    states = result_df['customer_state'].values
    x = np.arange(len(states))
    width = 0.18
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    for i, (comp, color) in enumerate(zip(comp_opp, colors)):
        vals = result_df[f'{comp}_norm'].values * w_star[i]
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"{comp} (w={w_star[i]:.2f})", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45)
    ax.legend()
    plt.title("EPS Component Contributions by State")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig1_component_bar.png")
    plt.close()

def run_scoring_pipeline(features_path: Path, pred_path: Path, output_dir: Path, fig_dir: Path, config: Dict[str, Any]) -> None:
    """
    Executes the full EPS scoring pipeline.
    """
    print_section_header("STARTING EPS PIPELINE")
    
    comp_opp = ['PD', 'GP', 'PG', 'MMI']
    constraints = {k: tuple(v) for k, v in config['scoring']['constraints'].items()}

    # Pipeline
    df_state = load_and_prepare_data(features_path, pred_path, config['n_weeks'])
    raw_df = calculate_raw_components(df_state, config['min_sellers'])
    norm_df = normalize_components(raw_df, comp_opp)
    w_star = find_optimal_weights(norm_df, comp_opp, constraints)
    result_df = compute_eps_scores(norm_df, raw_df, w_star, config['gamma'], comp_opp)

    # Outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "eps_results.csv", index=False)
    with open(output_dir / "w_star.json", "w") as f:
        json.dump({"w_star": {c: float(w) for c, w in zip(comp_opp, w_star)}, "gamma": config['gamma']}, f, indent=2)
    
    plot_results(result_df, w_star, comp_opp, fig_dir)
    
    print_section_header("TOP 5 EXPANSION TARGETS")
    logger.info("\n" + result_df[['EPS_rank', 'customer_state', 'EPS_score']].head(5).to_string(index=False))
    print_section_header("PIPELINE COMPLETED")
