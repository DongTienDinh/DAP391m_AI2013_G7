import sys
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src.core.config import get_config

cfg = get_config()
P = cfg.paths

EPS_PATH = P.outputs.eps_results
W_STAR_PATH = P.outputs.w_star
GEO_PATH = P.data.external_geojson
FIGURES_DIR = P.reports.figures_dir
OUTPUT_DIR = P.outputs.eps_dir

COMP_OPP = ["PD", "GP", "PG", "MMI"]
COMP_COLORS = {"PD": "#4C72B0", "GP": "#DD8452", "PG": "#55A868", "MMI": "#C44E52"}
CONSTRAINTS = {"PD": (0.25, 0.45), "GP": (0.15, 0.35), "PG": (0.15, 0.30), "MMI": (0.05, 0.15)}

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def load_data():
    df = pd.read_csv(EPS_PATH)
    with open(W_STAR_PATH) as f:
        config = json.load(f)
    w_star = np.array([config["w_star"][c] for c in COMP_OPP])
    gamma = config.get("gamma", 0.20)
    return df, w_star, gamma


def plot_component_contributions(df, w_star, gamma):
    """Weighted component bar chart + risk adjustment overlay."""
    fig, ax = plt.subplots(figsize=(14, 6))
    states = df["customer_state"].values
    x = np.arange(len(states))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for comp, offset in zip(COMP_OPP, offsets):
        vals = df[f"{comp}_norm"].values * w_star[COMP_OPP.index(comp)]
        ax.bar(x + offset * width, vals, width, label=f"{comp} (w={w_star[COMP_OPP.index(comp)]:.2f})",
               color=COMP_COLORS[comp], alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, df["Risk_Adj"].values, color="black", linewidth=1.5,
             linestyle="--", label=f"Risk Adjustment (1-{gamma}*LC)", marker="o", markersize=3)
    ax2.set_ylim(0.7, 1.05)
    ax2.set_ylabel("Risk Adjustment factor", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weighted component score", fontsize=10)
    ax.set_title("EPS: Component contributions by state (ranked by EPS score)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_component_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_component_bar.png")


def plot_choropleth(df):
    """3 side-by-side choropleth maps: EPS, OPP, LC."""
    if gpd is None or not GEO_PATH.exists():
        print("  ⚠ Skipping fig2_choropleth.png (geopandas or geojson missing)")
        return

    gdf_states = gpd.read_file(GEO_PATH)
    candidates = [c for c in gdf_states.columns if gdf_states[c].astype(str).str.len().max() == 2]
    state_key = candidates[0] if candidates else "sigla"
    gdf_states[state_key] = gdf_states[state_key].astype(str).str.upper().str.strip()

    res_copy = df.copy()
    res_copy["customer_state"] = res_copy["customer_state"].astype(str).str.upper().str.strip()
    gdf = gdf_states.merge(res_copy, left_on=state_key, right_on="customer_state", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    plot_configs = [
        ("EPS_score", "EPS Score (0–100)", "YlOrRd"),
        ("OPP_score", "Opportunity Score (OPP)", "Blues"),
        ("LC_norm", "Logistics Cost (normalised)", "Reds"),
    ]
    for ax, (col, title, cmap) in zip(axes, plot_configs):
        gdf.plot(column=col, ax=ax, cmap=cmap, legend=True,
                 legend_kwds={"label": title, "orientation": "horizontal", "shrink": 0.7},
                 missing_kwds={"color": "lightgrey", "label": "No data"},
                 edgecolor="white", linewidth=0.4)
        for _, row in gdf.iterrows():
            if pd.notnull(row.get(col)) and row.geometry is not None:
                try:
                    centroid = row.geometry.centroid
                    ax.annotate(text=row[state_key], xy=(centroid.x, centroid.y),
                                ha="center", va="center", fontsize=5.5, fontweight="bold", color="#333333")
                except Exception:
                    pass
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")

    plt.suptitle("EPS: Spatial distribution across Brazilian states", fontsize=13, y=1.01)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig2_choropleth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_choropleth.png")


def plot_radar_profiles(df, top_n=6):
    """Radar profiles for top N states."""
    top_states = df.sort_values("EPS_rank").head(top_n)
    labels = ["PD", "GP", "PG", "MMI", "Logistics\nQuality"]
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist() + [0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
    fig.suptitle("EPS: Component radar profiles — Top 6 states", fontsize=13)
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_states.iterrows()):
        ax = axes[i]
        state = row["customer_state"]
        vals = [row[f"{c}_norm"] for c in COMP_OPP] + [1.0 - row["LC_norm"]]
        vals += vals[:1]

        ax.plot(angles, vals, color="steelblue", linewidth=2)
        ax.fill(angles, vals, color="steelblue", alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6, color="grey")
        ax.set_title(f"Rank #{int(row['EPS_rank'])}: {state}  EPS={row['EPS_score']:.1f}", size=10, weight="bold", pad=14)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig3_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_radar.png")


def plot_correlation_heatmap(df):
    """Pearson correlation heatmap of components and final scores."""
    corr_cols = ["PD_norm", "GP_norm", "PG_norm", "MMI_norm", "LC_norm", "OPP_score", "EPS_score"]
    existing = [c for c in corr_cols if c in df.columns]
    if len(existing) < 2:
        print("  ⚠ Skipping correlation heatmap (insufficient columns)")
        return

    corr_matrix = df[existing].corr(method="pearson")
    rename = {"PD_norm": "PD (Demand)", "GP_norm": "GP (Growth)", "PG_norm": "PG (Penetration)",
              "MMI_norm": "MMI (Momentum)", "LC_norm": "LC (Logistics Cost)",
              "OPP_score": "OPP (Pre-risk)", "EPS_score": "EPS (Final)"}
    active = {k: v for k, v in rename.items() if k in existing}
    corr_matrix = corr_matrix.rename(columns=active, index=active)

    sns.set_theme(style="white")
    fig = plt.figure(figsize=(9, 7))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1.0, vmax=1.0,
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"})
    plt.title("Correlation Heatmap of EPS Components & Final Scores", fontsize=12, pad=15)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig3b_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3b_correlation_heatmap.png")


# ── Sensitivity Helpers ───────────────────────────────────────────────────────

def compute_eps_from_weights(w, norm_df, gamma):
    """Compute EPS scores from a weight vector and normalized data."""
    opp = (norm_df[COMP_OPP].values * w).sum(axis=1)
    eps_raw = opp * (1.0 - gamma * norm_df["LC"].values)
    eps_min, eps_max = eps_raw.min(), eps_raw.max()
    return (eps_raw - eps_min) / (eps_max - eps_min + 1e-9) * 100.0


def plot_monte_carlo(df, w_star, gamma, n_sim=10000, seed=42):
    """Monte Carlo: perturb weights, histogram of Spearman rank correlations."""
    print("  → Monte Carlo simulation...")
    rng = np.random.default_rng(seed)
    bounds_lo = np.array([CONSTRAINTS[c][0] for c in COMP_OPP])
    bounds_hi = np.array([CONSTRAINTS[c][1] for c in COMP_OPP])
    rank_base = df["EPS_rank"].values
    rho_list = []

    for _ in range(n_sim):
        w_sim = rng.dirichlet(w_star * 50)
        w_sim = np.clip(w_sim, bounds_lo, bounds_hi)
        w_sim /= w_sim.sum()
        eps_sim = compute_eps_from_weights(w_sim, df, gamma)
        rank_sim = pd.Series(eps_sim).rank(ascending=False).values
        rho, _ = spearmanr(rank_base, rank_sim)
        rho_list.append(rho)

    rho_arr = np.array(rho_list)
    mean_rho = rho_arr.mean()
    pct_095 = (rho_arr > 0.95).mean() * 100
    verdict = "ROBUST" if pct_095 >= 95 else ("MODERATE" if (rho_arr > 0.90).mean() * 100 >= 80 else "SENSITIVE")
    print(f"     Mean ρ={mean_rho:.4f}, >0.95={pct_095:.0f}%, verdict={verdict}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rho_arr, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.95, color="darkred", linestyle="--", label="ρ=0.95 (ROBUST)")
    ax.axvline(0.90, color="orange", linestyle="--", label="ρ=0.90 (MODERATE)")
    ax.axvline(mean_rho, color="green", linestyle="-", linewidth=1.8, label=f"Mean ρ={mean_rho:.4f}")
    ax.set_xlabel("Spearman ρ (rank correlation vs baseline)")
    ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo sensitivity (n={n_sim:,} simulations)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_monte_carlo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_monte_carlo.png")


def plot_oat_sweep(df, w_star, gamma, n_steps=40):
    """OAT sweep: vary each weight, track Spearman ρ."""
    print("  → OAT weight sweep...")
    rank_base = df["EPS_rank"].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("OAT Weight Sweep — Spearman ρ vs weight value", fontsize=12)
    axes = axes.flatten()

    for i, comp in enumerate(COMP_OPP):
        lo, hi = CONSTRAINTS[comp]
        sweep = np.linspace(lo, hi, n_steps)
        rho_list = []

        for w_val in sweep:
            w_new = w_star.copy()
            w_new[i] = w_val
            others = [j for j in range(len(COMP_OPP)) if j != i]
            s_others = w_new[others].sum()
            if s_others > 0:
                w_new[others] *= (1.0 - w_val) / s_others

            eps_sim = compute_eps_from_weights(w_new, df, gamma)
            rank_sim = pd.Series(eps_sim).rank(ascending=False).values
            rho, _ = spearmanr(rank_base, rank_sim)
            rho_list.append(rho)

        min_rho = min(rho_list)
        ax = axes[i]
        ax.plot(sweep, rho_list, marker="o", markersize=3, linestyle="-", color="teal")
        ax.axhline(0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="ρ=0.90")
        ax.axvline(w_star[i], color="green", linestyle=":", linewidth=1.2, label=f"w*={w_star[i]:.3f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(0.80, min_rho - 0.02), 1.02)
        ax.set_title(f"w({comp})  range=[{lo},{hi}]", fontsize=10)
        ax.set_xlabel(f"w_{comp}")
        ax.set_ylabel("Spearman ρ")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_oat_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig5_oat_sweep.png")


def plot_gamma_sweep(df, w_star, gamma):
    """Gamma sweep: vary risk penalty, track ρ and rank shifts."""
    print("  → Gamma sweep...")
    rank_base = df["EPS_rank"].values
    gamma_range = np.linspace(0.05, 0.40, 30)
    rho_gamma = []
    rank_shifts = []

    for g in gamma_range:
        eps_g = compute_eps_from_weights(w_star, df, gamma=g)
        rank_g = pd.Series(eps_g).rank(ascending=False).values
        rho, _ = spearmanr(rank_base, rank_g)
        rho_gamma.append(rho)
        rank_shifts.append((np.abs(rank_g - rank_base) >= 3).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Gamma (risk weight) sweep", fontsize=12)

    ax1.plot(gamma_range, rho_gamma, color="steelblue", linewidth=2, marker="o", markersize=3)
    ax1.axvline(gamma, color="green", linestyle="--", linewidth=1.5, label=f"γ={gamma} (current)")
    ax1.axhline(0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="ρ=0.90")
    ax1.set_xlabel("γ (risk penalty weight)")
    ax1.set_ylabel("Spearman ρ vs baseline")
    ax1.set_title("Rank stability across γ values")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(gamma_range, rank_shifts, color="coral", linewidth=2, marker="s", markersize=3)
    ax2.axvline(gamma, color="green", linestyle="--", linewidth=1.5, label=f"γ={gamma} (current)")
    ax2.set_xlabel("γ (risk penalty weight)")
    ax2.set_ylabel("# states with |Δrank| ≥ 3")
    ax2.set_title("States shifting ≥3 positions")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_gamma_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig6_gamma_sweep.png")


def main():
    print("=" * 60)
    print("  Generating static report figures")
    print("=" * 60)

    if not EPS_PATH.exists():
        print(f"  ✗ EPS results not found at {EPS_PATH}")
        print("  Run scoring pipeline first.")
        return

    df, w_star, gamma = load_data()
    print(f"  Loaded {len(df)} states, weights: {dict(zip(COMP_OPP, [round(w, 4) for w in w_star]))}")

    plot_component_contributions(df, w_star, gamma)
    plot_choropleth(df)
    plot_radar_profiles(df)
    plot_correlation_heatmap(df)

    plot_monte_carlo(df, w_star, gamma)
    plot_oat_sweep(df, w_star, gamma)
    plot_gamma_sweep(df, w_star, gamma)

    print(f"\n  ✅ All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
