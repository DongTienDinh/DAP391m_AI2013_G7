#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ranking Comparison: EPS-final vs Progressive Baselines
=======================================================
Computes 4 progressive baselines and evaluates against EPS-final
using Spearman correlation, Top-5 overlap, and rank shift metrics.

Baselines:
  B1 — Revenue-only:      rank by last-week revenue
  B2 — Forecast-only:     rank by ML-predicted next-week revenue
  B3 — OPP-only:          rank by weighted opportunity score (no logistics penalty)
  B4 — LA-Revenue:        rank by revenue_rolling_8 × (1 − γ·LC_norm)
                          (8-week rolling mean; avoids zero-revenue collapse
                           in sparse states — Hyndman & Athanasopoulos, 2021)

Paper references:
  - Spearman (1904) for rank correlation
  - Manning et al. (2008) for Precision@k
  - Saltelli et al. (2004) for rank sensitivity analysis
  - OECD/JRC (2008) Handbook on Constructing Composite Indicators, Section 6
  - Hyndman & Athanasopoulos (2021) FPP3 §2.4 — rolling mean for demand smoothing
  - Box et al. (2015) Time Series Analysis — smoothing prior to feature use
  - Chopra & Meindl (2016) Supply Chain Management §7 — regional demand estimation

Usage:
    python src/analysis/ranking_comparison.py
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Configure utf-8 encoding for stdout on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Path resolution ──────────────────────────────────────────────────────────
# Script is at: <project_root>/src/analysis/ranking_comparison.py
# Project root: <project_root>/
ROOT = Path(__file__).resolve().parents[2]

# Expected file paths
FEATURES_PATH = ROOT / "data" / "processed" / "olist" / "features_weekly.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "olist" / "predicted_next_week_revenue.csv"
EPS_RESULTS_PATH = ROOT / "outputs" / "eps" / "eps_results.csv"
OUTPUT_CSV = ROOT / "reports" / "ranking_comparison.csv"
OUTPUT_TXT = ROOT / "reports" / "ranking_comparison_summary.txt"

# EPS weight constants (from SLSQP entropy-maximization optimization)
W_PD = 0.2939
W_GP = 0.2561
W_PG = 0.3000
W_MMI = 0.1500
# Fixed heuristic coefficient, not SLSQP-optimized (see expansion_scoring.py for EPS weight optimization)
GAMMA = 0.20

# All 27 Brazilian states
BR_STATES = {
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO',
    'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR',
    'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO',
}
LAST_WEEK = "2018-08-20/2018-08-26"

# Window for B4 smoothed revenue signal.
# 8-week rolling mean mitigates single-period demand volatility in sparse states
# (Hyndman & Athanasopoulos, 2021, FPP3 §2.4; Chopra & Meindl, 2016, §7).
ROLLING_WINDOW = "revenue_rolling_8"


# ── Helper Functions ─────────────────────────────────────────────────────────

def normalize_minmax(series: pd.Series, scale: float = 100.0) -> pd.Series:
    """
    Min-max normalization to [0, scale].

    Uses a small epsilon in the denominator to avoid division by zero
    when all values are identical.
    """
    eps_val = 1e-8
    return (series - series.min()) / (series.max() - series.min() + eps_val) * scale


def _resolve_path(primary: Path, label: str) -> Path:
    """
    Resolve a file path. If the primary path does not exist, search
    recursively from ROOT. Raises FileNotFoundError if not found anywhere.
    """
    if primary.exists():
        return primary

    print(f"  Warning: {label} not found at {primary}")
    print(f"  Searching recursively under {ROOT} ...")
    candidates = list(ROOT.rglob(primary.name))
    if candidates:
        chosen = candidates[0]
        print(f"  Found: {chosen}")
        return chosen

    raise FileNotFoundError(
        f"{label} not found at {primary} or anywhere under {ROOT}"
    )


def _validate_states(df: pd.DataFrame, file_label: str) -> None:
    """
    Validate that all 27 Brazilian states are present. Print a warning
    for any missing states.
    """
    present = set(df['customer_state'].unique())
    missing = BR_STATES - present
    if missing:
        warnings.warn(
            f"[{file_label}] Missing {len(missing)} state(s): "
            f"{sorted(missing)}"
        )
    else:
        print(f"  ✓ All 27 states present in {file_label}")


# ── Core Functions ───────────────────────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate all three input files:
      1. eps_results.csv — EPS final scores and component norms
      2. features_weekly.csv — filtered to last week
      3. predicted_next_week_revenue.csv — one row per state
    """
    print("=" * 60)
    print("  Loading input data")
    print("=" * 60)

    # 1. EPS results
    eps_path = _resolve_path(EPS_RESULTS_PATH, "eps_results.csv")
    df_eps = pd.read_csv(eps_path)
    print(f"  Loaded eps_results.csv: {len(df_eps)} rows")
    _validate_states(df_eps, "eps_results.csv")

    # 2. Weekly features — filter to last week
    feat_path = _resolve_path(FEATURES_PATH, "features_weekly.csv")
    df_features_all = pd.read_csv(feat_path)
    df_features_last = df_features_all[
        df_features_all['year_week'] == LAST_WEEK
    ].copy()
    print(f"  Loaded features_weekly.csv: {len(df_features_all)} total rows, "
          f"{len(df_features_last)} rows for week {LAST_WEEK}")
    _validate_states(df_features_last, f"features_weekly.csv (week={LAST_WEEK})")

    # 3. Predictions
    pred_path = _resolve_path(PREDICTIONS_PATH, "predicted_next_week_revenue.csv")
    df_forecast = pd.read_csv(pred_path)
    print(f"  Loaded predicted_next_week_revenue.csv: {len(df_forecast)} rows")
    _validate_states(df_forecast, "predicted_next_week_revenue.csv")

    return df_eps, df_features_last, df_forecast


def compute_baselines(
    df_eps: pd.DataFrame,
    df_features_last: pd.DataFrame,
    df_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all 4 baselines and merge into a single per-state DataFrame.

    Baselines:
      B1 — Revenue-only:      normalize_minmax(revenue_last_week)
      B2 — Forecast-only:     normalize_minmax(predicted_next_week_revenue)
      B3 — OPP-only:          normalize_minmax(w·PD + w·GP + w·PG + w·MMI)
      B4 — LA-Revenue:        normalize_minmax(revenue × (1 − γ·LC_norm))

    All ranks use method='min', ascending=False (highest score = rank 1).
    """
    print("\n" + "=" * 60)
    print("  Computing baselines")
    print("=" * 60)

    # Start with EPS reference columns
    df = df_eps[['customer_state', 'EPS_score', 'EPS_rank',
                 'PD_norm', 'GP_norm', 'PG_norm', 'MMI_norm', 'LC_norm']].copy()

    # Merge last-week revenue
    rev = df_features_last[['customer_state', 'revenue', 'revenue_rolling_8']].copy()
    rev = rev.rename(columns={'revenue': 'revenue_last_week'})
    df = df.merge(rev, on='customer_state', how='left')

    # Merge predicted revenue
    pred = df_forecast[['customer_state', 'predicted_next_week_revenue']].copy()
    pred = pred.rename(columns={'predicted_next_week_revenue': 'predicted_revenue'})
    df = df.merge(pred, on='customer_state', how='left')

    # ── Baseline 1: Revenue-only ──
    # Research Q: How different is EPS from ranking purely by current revenue?
    # Captures only current demand level with no growth, penetration, or logistics signal.
    df['score_revenue'] = normalize_minmax(df['revenue_last_week'])
    df['rank_revenue'] = df['score_revenue'].rank(ascending=False, method='min').astype(int)

    # ── Baseline 2: Forecast-only ──
    # Research Q: How different is EPS from ranking purely by ML-predicted next-week revenue?
    # Captures future demand signal but ignores penetration gap, seller momentum, and logistics risk.
    df['score_forecast'] = normalize_minmax(df['predicted_revenue'])
    df['rank_forecast'] = df['score_forecast'].rank(ascending=False, method='min').astype(int)

    # ── Baseline 3: OPP-only (no logistics penalty) ──
    # Research Q: What is the marginal contribution of the logistics penalty?
    # γ is set to 0.20 as a fixed business-rule coefficient (not optimized);
    # its purpose is to represent a simple heuristic penalty, not an entropy-optimal weight.
    opp_raw = (
        W_PD  * df['PD_norm'] +
        W_GP  * df['GP_norm'] +
        W_PG  * df['PG_norm'] +
        W_MMI * df['MMI_norm']
    )
    df['score_opp'] = normalize_minmax(opp_raw)
    df['rank_opp'] = df['score_opp'].rank(ascending=False, method='min').astype(int)

    # ── Baseline 4: Logistics-adjusted revenue (smoothed) ──
    # Uses 8-week rolling mean revenue instead of single-week snapshot.
    # Rationale: 8/27 states have revenue=0 in the last observed week due to
    # Olist dataset sparsity. A single-week signal of 0 causes la_raw=0 regardless
    # of LC_norm, making the logistics penalty ineffective and producing rankings
    # identical to B1 (Revenue-only). The 8-week rolling mean is a standard
    # noise-reduction technique for sparse demand signals
    # (Hyndman & Athanasopoulos, 2021; Box et al., 2015).
    # B1 retains raw revenue_last_week to preserve its role as a pure
    # point-in-time revenue baseline.
    la_raw = df[ROLLING_WINDOW] * (1.0 - GAMMA * df['LC_norm'])
    df['score_la'] = normalize_minmax(la_raw)
    df['rank_la'] = df['score_la'].rank(ascending=False, method='min').astype(int)

    # ── Per-state rank deltas ──
    # Positive delta = EPS promotes this state above the baseline ranking
    # Negative delta = EPS demotes this state relative to the baseline
    df['delta_revenue']  = df['rank_revenue']  - df['EPS_rank']
    df['delta_forecast'] = df['rank_forecast'] - df['EPS_rank']
    df['delta_opp']      = df['rank_opp']      - df['EPS_rank']
    df['delta_la']       = df['rank_la']        - df['EPS_rank']

    # Sort by EPS rank ascending
    df = df.sort_values('EPS_rank').reset_index(drop=True)

    print(f"  ✓ Computed all 4 baselines for {len(df)} states")
    return df


def compute_metrics(
    df: pd.DataFrame,
    rank_col: str,
    delta_col: str,
    eps_rank_col: str = 'EPS_rank',
) -> Dict:
    """
    Compute evaluation metrics comparing a baseline ranking against EPS-final.

    Metrics:
      1. Spearman rank correlation (ρ, p-value)
      2. Top-5 overlap (Precision@5)
      3. Maximum absolute rank shift + argmax state
      4. Mean absolute rank shift
    """
    rank_eps = df[eps_rank_col].values
    rank_baseline = df[rank_col].values
    delta = df[delta_col].values

    # 1. Spearman rank correlation
    rho, pval = spearmanr(rank_eps, rank_baseline)

    # 2. Top-5 overlap (Precision@5)
    top5_eps = set(df.loc[df[eps_rank_col] <= 5, 'customer_state'])
    top5_baseline = set(df.loc[df[rank_col] <= 5, 'customer_state'])
    overlap = len(top5_eps & top5_baseline)
    overlap_rate = overlap / 5.0

    # 3. Maximum absolute rank shift
    abs_delta = np.abs(delta)
    max_shift = int(abs_delta.max())
    max_shift_idx = abs_delta.argmax()
    max_shift_state = df.iloc[max_shift_idx]['customer_state']

    # 4. Mean absolute rank shift
    mean_shift = float(abs_delta.mean())

    return {
        'spearman_rho': rho,
        'spearman_pval': pval,
        'top5_overlap': overlap,
        'top5_overlap_rate': overlap_rate,
        'top5_states': sorted(top5_baseline),
        'max_shift': max_shift,
        'max_shift_state': max_shift_state,
        'mean_shift': mean_shift,
    }


def format_summary(df: pd.DataFrame, all_metrics: Dict[str, Dict]) -> str:
    """
    Format the summary metrics table as a readable string for stdout and .txt output.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  Ranking Comparison Summary: EPS-final vs Progressive Baselines")
    lines.append("=" * 80)
    lines.append("")

    # Column headers
    baselines = ['Revenue-only', 'Forecast-only', 'OPP-only', 'LA-Revenue']
    keys = ['revenue', 'forecast', 'opp', 'la']

    header = f"{'Metric':<24s}"
    for b in baselines:
        header += f"  {b:>14s}"
    lines.append(header)
    lines.append("-" * len(header))

    # Spearman rho
    row = f"{'Spearman rho':<24s}"
    for k in keys:
        row += f"  {all_metrics[k]['spearman_rho']:>14.4f}"
    lines.append(row)

    # Spearman p-value
    row = f"{'Spearman p-value':<24s}"
    for k in keys:
        row += f"  {all_metrics[k]['spearman_pval']:>14.4f}"
    lines.append(row)

    # Top-5 overlap
    row = f"{'Top-5 overlap':<24s}"
    for k in keys:
        row += f"  {all_metrics[k]['top5_overlap']:>12d}/5"
    lines.append(row)

    # Top-5 overlap rate
    row = f"{'Top-5 overlap rate':<24s}"
    for k in keys:
        row += f"  {all_metrics[k]['top5_overlap_rate']:>14.2f}"
    lines.append(row)

    # Max rank shift
    row = f"{'Max rank shift':<24s}"
    for k in keys:
        m = all_metrics[k]
        val = f"{m['max_shift']} ({m['max_shift_state']})"
        row += f"  {val:>14s}"
    lines.append(row)

    # Mean rank shift
    row = f"{'Mean rank shift':<24s}"
    for k in keys:
        row += f"  {all_metrics[k]['mean_shift']:>14.2f}"
    lines.append(row)

    lines.append("")

    # Top-5 lists
    top5_eps = sorted(
        df.loc[df['EPS_rank'] <= 5, 'customer_state'].tolist()
    )
    lines.append(f"Top-5 EPS states:     {top5_eps}")
    for k, label in zip(keys, baselines):
        lines.append(f"Top-5 {label + ':':<17s} {all_metrics[k]['top5_states']}")

    lines.append("")

    # Notable rank shifts (|delta| >= 10)
    lines.append("Notable rank shifts (|delta| >= 10):")
    rank_cols = [
        ('delta_revenue',  'rank_revenue',  'Revenue-only'),
        ('delta_forecast', 'rank_forecast', 'Forecast-only'),
        ('delta_opp',      'rank_opp',      'OPP-only'),
        ('delta_la',       'rank_la',       'LA-Revenue'),
    ]
    found_notable = False
    for delta_col, rank_col, label in rank_cols:
        notable = df[df[delta_col].abs() >= 10]
        for _, row in notable.iterrows():
            found_notable = True
            lines.append(
                f"  {row['customer_state']}: EPS_rank={int(row['EPS_rank'])}, "
                f"{label}_rank={int(row[rank_col])}, "
                f"delta={int(row[delta_col])}"
            )
    if not found_notable:
        lines.append("  (none)")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    """
    Main entry point:
      1. Load all input data
      2. Compute 4 progressive baselines
      3. Evaluate each baseline against EPS-final
      4. Save per-state comparison CSV
      5. Print + save summary metrics
    """
    # 1. Load data
    df_eps, df_features_last, df_forecast = load_data()

    # 2. Compute all baselines
    df = compute_baselines(df_eps, df_features_last, df_forecast)

    # 3. Compute evaluation metrics for each baseline
    print("\n" + "=" * 60)
    print("  Computing evaluation metrics")
    print("=" * 60)

    baseline_configs = {
        'revenue':  ('rank_revenue',  'delta_revenue'),
        'forecast': ('rank_forecast', 'delta_forecast'),
        'opp':      ('rank_opp',      'delta_opp'),
        'la':       ('rank_la',       'delta_la'),
    }

    all_metrics = {}
    for key, (rank_col, delta_col) in baseline_configs.items():
        metrics = compute_metrics(df, rank_col, delta_col)
        all_metrics[key] = metrics
        print(f"  ✓ {key}: ρ={metrics['spearman_rho']:.4f}, "
              f"Top-5={metrics['top5_overlap']}/5, "
              f"MaxShift={metrics['max_shift']} ({metrics['max_shift_state']}), "
              f"MeanShift={metrics['mean_shift']:.2f}")

    # 4. Save per-state comparison CSV
    output_cols = [
        'customer_state',
        'EPS_score', 'EPS_rank',
        'score_revenue', 'rank_revenue', 'delta_revenue',
        'score_forecast', 'rank_forecast', 'delta_forecast',
        'score_opp', 'rank_opp', 'delta_opp',
        'score_la', 'rank_la', 'delta_la',
        'revenue_last_week', 'revenue_rolling_8', 'predicted_revenue',
    ]
    df_out = df[output_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Saved ranking comparison table to: {OUTPUT_CSV}")

    # 5. Print + save summary
    summary = format_summary(df, all_metrics)
    print("\n" + summary)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  ✓ Saved summary to: {OUTPUT_TXT}")

    # Sanity check: B1 and B4 must now produce different rankings
    b1_b4_identical = (df['rank_revenue'] == df['rank_la']).all()
    if b1_b4_identical:
        raise ValueError(
            "B1 and B4 rankings are identical — revenue signal for B4 may need review."
        )
    else:
        n_diff = (df['rank_revenue'] != df['rank_la']).sum()
        print(f"  ✓ B1 vs B4: rankings differ for {n_diff}/27 states (expected)")


if __name__ == '__main__':
    main()
