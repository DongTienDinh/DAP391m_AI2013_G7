import sys
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

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

    print(f"\n  ✅ All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
