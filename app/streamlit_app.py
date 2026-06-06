#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Application for E-Commerce Expansion Priority Score (EPS) Dashboard

This dashboard visualizes state-level rankings, components contributions,
spatial maps, and sensitivity analysis, with real-time recalculation and
COMPASS-XAI explanations.
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Declare custom map component
parent_dir = Path(__file__).resolve().parent
component_dir = parent_dir / "map_component"
map_selector = components.declare_component("map_selector", path=str(component_dir))
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Resolve project root path and insert into sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import geopandas and ignore warning if not available
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# Load environment variables
load_dotenv(Path.cwd() / ".env")

# Import XAI explainer functions
from src.olist_pipeline.analysis.xai import format_narrative, call_gemini_narrative, assign_tier

# ── State and Indicator Mapping Dictionaries ───────────────────────────────
STATE_MAP = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
    "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins"
}

METRIC_NAMES = {
    "PD": "Predicted Demand",
    "GP": "Growth Potential",
    "PG": "Penetration Gap",
    "MMI": "Market Momentum Index"
}

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Expansion Priority Score (EPS) Dashboard",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Dark Mode CSS ──
st.markdown("""
<style>
    /* Hide Streamlit default header, deploy button, and hamburger menu */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}

    /* Global Dark Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important; /* Slate 900 */
        color: #f8fafc !important;
    }
    /* Override dark mode text */
    h1, h2, h3, h4, h5, h6, p, span, label, .markdown-text-container {
        color: #e2e8f0 !important;
    }
    /* Dark Mode Ranking Card */
    .eps-ranking-card {
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
    .eps-ranking-card h3, .eps-ranking-card p, .eps-ranking-card span {
        color: #f8fafc !important;
    }
    .eps-ranking-card div {
        background-color: #334155 !important;
        border-color: #475569 !important;
    }
    /* Make metric cards readable in dark mode */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Disable the annoying 'c' hotkey that opens the "Clear caches" modal
components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'c' || e.key === 'C') {
            // Stop propagation to prevent Streamlit from catching it, 
            // but DO NOT preventDefault so native Ctrl+C copy still works.
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, true);
    </script>
    """,
    height=0, width=0,
)

from src.olist_pipeline.utils.config_loader import Config

# ... rest of imports ...

# ── Data Loading & Helper Functions ───────────────────────────────────────────
@st.cache_data
def load_base_data() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Loads results and configurations from disk.
    """
    eps_path = Config.get_path("outputs", "eps_results")
    w_star_path = Config.get_path("outputs", "w_star")
    report_json_path = Config.get_path("outputs", "xai_report_json")

    if not eps_path.exists() or not w_star_path.exists() or not report_json_path.exists():
        st.error("Error: Output results not found. Make sure to run the main scoring and explainer pipeline first.")
        st.stop()

    df = pd.read_csv(eps_path)
    with open(w_star_path, "r", encoding="utf-8") as f:
        w_config = json.load(f)
    with open(report_json_path, "r", encoding="utf-8") as f:
        xai_report = json.load(f)

    return df, w_config, xai_report


# Recalculate EPS based on custom weights
def recalculate_eps(
    df: pd.DataFrame,
    w_pd: float,
    w_gp: float,
    w_pg: float,
    w_mmi: float,
    gamma: float
) -> pd.DataFrame:
    """
    Recalculates EPS scores and ranks in-memory.
    """
    res = df.copy()
    
    # Calculate Opportunity score (weighted sum of normalized components)
    opp_score = (
        res['PD_norm'] * w_pd +
        res['GP_norm'] * w_gp +
        res['PG_norm'] * w_pg +
        res['MMI_norm'] * w_mmi
    )
    
    # Logistics risk adjustment factor
    risk_adj = 1.0 - gamma * res['LC_norm']
    eps_raw = opp_score * risk_adj
    
    # Rescale raw EPS to [0, 100]
    eps_min = eps_raw.min()
    eps_max = eps_raw.max()
    eps_score = (eps_raw - eps_min) / (eps_max - eps_min + 1e-9) * 100.0
    
    res['EPS_score'] = eps_score
    res['OPP_score'] = opp_score
    res['Risk_Adj'] = risk_adj
    res['EPS_rank'] = pd.Series(eps_score).rank(ascending=False).astype(int).values
    
    # Recalculate component absolute and percentage contributions
    COMP = ['PD', 'GP', 'PG', 'MMI']
    weights = {'PD': w_pd, 'GP': w_gp, 'PG': w_pg, 'MMI': w_mmi}
    for c in COMP:
        res[f'contrib_{c}'] = weights[c] * res[f'{c}_norm']
        
    res['risk_penalty_abs'] = res['OPP_score'] * gamma * res['LC_norm']
    res['risk_penalty_pct'] = gamma * res['LC_norm'] * 100
    
    for c in COMP:
        res[f'contrib_pct_{c}'] = np.where(
            res['OPP_score'] > 1e-9,
            (res[f'contrib_{c}'] / res['OPP_score']) * 100,
            0.0
        )
        
    res['dominant_component'] = res[[f'contrib_{c}' for c in COMP]].idxmax(axis=1).str.replace('contrib_', '')
    res['weakest_component'] = res[[f'contrib_{c}' for c in COMP]].idxmin(axis=1).str.replace('contrib_', '')
    
    # Recalculate priority tier
    res['tier'] = res['EPS_rank'].apply(lambda r: assign_tier(r, len(res)))
    
    # Re-sort by recalculated rank
    return res.sort_values('EPS_rank').reset_index(drop=True)


# Render a dynamic choropleth map using GeoPandas
def plot_dynamic_map(df_recalc: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Loads geojson and renders dynamic matplotlib figure.
    """
    if gpd is None:
        return None
        
    geo_path = Config.get_path("data", "external_geojson")
    if not geo_path.exists():
        return None
        
    try:
        gdf_states = gpd.read_file(geo_path)
        # Find abbreviation column (size 2)
        candidates = [c for c in gdf_states.columns if gdf_states[c].astype(str).str.len().max() == 2]
        state_key = candidates[0] if candidates else 'sigla'
        
        gdf_states[state_key] = gdf_states[state_key].astype(str).str.upper().str.strip()
        
        res_copy = df_recalc.copy()
        res_copy['customer_state'] = res_copy['customer_state'].astype(str).str.upper().str.strip()
        
        gdf = gdf_states.merge(res_copy, left_on=state_key, right_on='customer_state', how='left')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Match slate-900 theme background
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        
        gdf.plot(
            column='EPS_score', ax=ax, cmap='YlOrRd',
            legend=True,
            legend_kwds={'label': 'Recalculated EPS Score', 'orientation': 'horizontal', 'shrink': 0.7, 'pad': 0.05},
            missing_kwds={'color': '#1e293b', 'label': 'No data'},
            edgecolor='#1e293b', linewidth=0.5
        )
        
        # Annotate state abbreviations
        for _, row in gdf.iterrows():
            if pd.notnull(row.get('EPS_score')) and row.geometry is not None:
                try:
                    centroid = row.geometry.centroid
                    ax.annotate(
                        text=row[state_key],
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center',
                        fontsize=6, fontweight='bold', color='#ffffff',
                        alpha=0.85
                    )
                except Exception:
                    pass
                    
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.sidebar.error(f"Failed to plot dynamic map: {e}")
        return None


# Render a radar chart for a single state
def plot_single_state_radar(state_row: pd.Series, color: str = '#3b82f6') -> plt.Figure:
    """
    Renders polar matplotlib radar chart for a single state.
    """
    labels = ['Predicted Demand\n(PD)', 'Growth Potential\n(GP)', 'Penetration Gap\n(PG)', 'Market Momentum\n(MMI)', 'Logistics Quality\n(1-LC)']
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#FFFFFF') # Match white theme container background
    ax.set_facecolor('#F8F9FA')        # Match light grid background
    
    vals = [
        float(state_row['PD_norm']),
        float(state_row['GP_norm']),
        float(state_row['PG_norm']),
        float(state_row['MMI_norm']),
        1.0 - float(state_row['LC_norm'])
    ]
    vals += vals[:1]
    
    ax.plot(angles, vals, color=color, linewidth=2.5)
    ax.fill(angles, vals, color=color, alpha=0.3)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8, color='#2C3E50')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=6, color='#64748B')
    
    # Style radial grid lines
    ax.grid(color='#E2E8F0', alpha=0.8)
    ax.spines['polar'].set_color('#CBD5E1')
    
    plt.tight_layout()
    return fig


def plot_combo_chart(df: pd.DataFrame, w_pd: float, w_gp: float, w_pg: float, w_mmi: float) -> go.Figure:
    """
    Creates a combination chart showing weighted component contributions (bars) 
    and the risk adjustment factor (line) across states.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Sort descending by EPS_score
    df_sorted = df.sort_values("EPS_score", ascending=False)
    x_labels = df_sorted['state_display']
    
    # Distinct pastel/muted palette for the components
    colors = {"PD": "#60a5fa", "GP": "#34d399", "PG": "#fbbf24", "MMI": "#a78bfa"}
    
    # Add grouped bars for component contributions
    fig.add_trace(go.Bar(
        x=x_labels, y=df_sorted['contrib_PD'], name=f"PD (w={w_pd:.2f})", marker_color=colors["PD"]
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=x_labels, y=df_sorted['contrib_GP'], name=f"GP (w={w_gp:.2f})", marker_color=colors["GP"]
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=x_labels, y=df_sorted['contrib_PG'], name=f"PG (w={w_pg:.2f})", marker_color=colors["PG"]
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=x_labels, y=df_sorted['contrib_MMI'], name=f"MMI (w={w_mmi:.2f})", marker_color=colors["MMI"]
    ), secondary_y=False)
    
    # Add secondary Y-axis line chart for Risk Adjustment
    fig.add_trace(go.Scatter(
        x=x_labels, y=df_sorted['Risk_Adj'], name="Risk Adjustment",
        mode='lines+markers', line=dict(color='#0f172a', dash='dash', width=2),
        marker=dict(size=6, color='#0f172a')
    ), secondary_y=True)
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2C3E50",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Weighted Component Score", secondary_y=False)
    fig.update_yaxes(title_text="Risk Adj (1 - γ*LC)", secondary_y=True, range=[0, 1.05])
    
    return fig


# Render an interactive Plotly map showing state borders, supporting hover and click
def draw_plotly_brazil_map(df_recalc: pd.DataFrame, selected_state: str, selected_state_2: Optional[str] = None, height: int = 520) -> go.Figure:
    """
    Renders an interactive Plotly MapLibre choropleth map highlighting selection.
    Uses a bright, light map style with teal-blue Brazil base color,
    vibrant selected state, crisp dark charcoal state borders,
    and 2-letter state abbreviation labels at centroids.
    """
    geo_path = Config.get_path("data", "external_geojson")
    
    # Load GeoJSON as raw dict for choropleth
    with open(geo_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    
    # Also read via geopandas for merging data
    gdf_states = gpd.read_file(geo_path)
    
    candidates = [c for c in gdf_states.columns if gdf_states[c].astype(str).str.len().max() == 2]
    state_key = candidates[0] if candidates else 'sigla'
    gdf_states[state_key] = gdf_states[state_key].astype(str).str.upper().str.strip()
    
    gdf = gdf_states.merge(df_recalc, left_on=state_key, right_on='customer_state', how='left')
    
    # Assign unique feature IDs in GeoJSON matching state codes
    for i, feature in enumerate(geojson_data["features"]):
        abbrev = feature["properties"].get("abbrev_state", "").upper().strip()
        feature["id"] = abbrev
    
    # ── Color Palette (Bright Theme) ──
    BRAZIL_BASE = "#7ecec1"        # Pleasant light teal-blue
    SELECTED_BLUE = "#2563eb"      # Vibrant saturated medium-blue
    COMPARISON_ORANGE = "#ea580c"  # Vibrant orange
    
    # Build color array based on selection
    colors = []
    customdata_list = []
    state_codes_list = []
    
    for _, row in gdf.iterrows():
        sc = row['customer_state'] if pd.notnull(row.get('customer_state')) else ''
        if sc == selected_state:
            colors.append(SELECTED_BLUE)
        elif selected_state_2 and sc == selected_state_2:
            colors.append(COMPARISON_ORANGE)
        else:
            colors.append(BRAZIL_BASE)
        
        state_name = f"{STATE_MAP.get(sc, sc)} ({sc})" if sc else "Unknown"
        eps_score = row['EPS_score'] if pd.notnull(row.get('EPS_score')) else 0
        eps_rank = row['EPS_rank'] if pd.notnull(row.get('EPS_rank')) else '-'
        tier = row['tier'] if pd.notnull(row.get('tier')) else '-'
        
        customdata_list.append([sc, state_name, eps_score, eps_rank, tier])
        state_codes_list.append(sc)
    
    # Use Choroplethmap (MapLibre, free — replaces deprecated Choroplethmapbox)
    fig = go.Figure(go.Choroplethmap(
        geojson=geojson_data,
        locations=state_codes_list,
        z=[1] * len(state_codes_list),
        colorscale=[[0, BRAZIL_BASE], [1, BRAZIL_BASE]],
        showscale=False,
        marker=dict(
            opacity=0.92,
            line=dict(width=2.0, color="#333333")  # High-contrast dark charcoal borders
        ),
        customdata=customdata_list,
        hovertemplate="<b>%{customdata[1]}</b><br>Priority Score (EPS): %{customdata[2]:.1f}/100<br>Rank: #%{customdata[3]} (%{customdata[4]} Tier)<extra></extra>",
        selectedpoints=[],
    ))
    
    # Override fill colors per-state using indexed colorscale
    fig.update_traces(
        marker_opacity=0.92,
        z=list(range(len(colors))),
        colorscale=[[i / max(len(colors) - 1, 1), c] for i, c in enumerate(colors)],
    )
    
    # ── State Abbreviation Labels ──
    centroids = gdf.geometry.centroid
    label_lats = [c.y for c in centroids]
    label_lons = [c.x for c in centroids]
    label_abbrevs = gdf[state_key].tolist()
    
    # Halo trace (white outline — larger font rendered behind main labels)
    fig.add_trace(go.Scattermap(
        lat=label_lats,
        lon=label_lons,
        mode="text",
        text=label_abbrevs,
        textfont=dict(
            size=14,
            color="white",
        ),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Main label trace (dark bold text on top of halo)
    fig.add_trace(go.Scattermap(
        lat=label_lats,
        lon=label_lons,
        mode="text",
        text=label_abbrevs,
        textfont=dict(
            size=11,
            color="#1a1a1a",
        ),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Bright light map with clear Brazil focus (uses MapLibre — no token needed)
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=-14.2, lon=-53.0),
            zoom=3.2,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
        dragmode="pan",
        clickmode="event+select",
        height=height,
    )
    
    return fig


# Render ranking card
def render_ranking_card(state_row: pd.Series, color: str = '#3b82f6'):
    # Large prominent Rank Display
    st.markdown(f"""
    <div class="eps-ranking-card" style="text-align: center; padding: 20px 0; background-color: #FFFFFF !important; border-radius: 8px; border: 1px solid #C5D3C1 !important; margin-bottom: 0px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h3 style="margin: 0; color: #64748B !important; text-transform: uppercase; font-size: 14px; letter-spacing: 2px; font-weight: 600; text-shadow: none !important;">EPS State Ranking</h3>
        <h1 style="font-size: 70px; margin: 5px 0; color: {color} !important; line-height: 1; font-weight: 800; text-shadow: none !important;">#{state_row['EPS_rank']}</h1>
        <div style="display: inline-block; padding: 4px 12px; background-color: #F8F9FA !important; border-radius: 20px; border: 1px solid #C5D3C1 !important;">
            <span style="color: #2C3E50 !important; font-weight: 700; font-size: 14px; text-shadow: none !important;">{state_row['tier']} PRIORITY TIER</span>
        </div>
        <p style="font-size: 16px; margin: 15px 0 0 0; color: #1F2937 !important; font-weight: 500; text-shadow: none !important;">EPS Score: <strong style="color: {color} !important; font-size: 20px;">{state_row['EPS_score']:.1f}</strong> / 100</p>
    </div>
    """, unsafe_allow_html=True)

# Render radar chart
def render_radar_chart_section(state_row: pd.Series, color: str = '#3b82f6'):
    st.markdown("#### :material/radar: Component Performance Profile")
    fig_radar = plot_single_state_radar(state_row, color)
    st.pyplot(fig_radar)

# Render warnings
def render_warnings(state_row: pd.Series):
    state_name = STATE_MAP.get(state_row['customer_state'], state_row['customer_state'])
    is_mmi_imputed = bool(np.abs(state_row['MMI_norm'] - np.median(df_recalc['MMI_norm'].values)) < 1e-4)
    if state_row['data_sparse']:
        st.warning(f"Data Sparse in {state_name}: Very low historical revenue. Interpretation of Growth Potential (GP) may be volatile.", icon=":material/warning:")
    if state_row['PG_norm'] < 0.1:
        st.info(f"Market Saturation in {state_name}: Penetration Gap is near zero, suggesting a mature e-commerce region.", icon=":material/info:")
    if state_row['LC_norm'] > 0.7:
        st.error(f"High Logistics Cost in {state_name}: Extreme transport expenses penalize priority ranking.", icon=":material/local_shipping:")
    if is_mmi_imputed:
        st.info(f"MMI Imputed in {state_name}: Market Momentum was filled using median due to low active sellers count.", icon=":material/hourglass_empty:")

# Render indicator scores
def render_indicator_scores(state_row: pd.Series):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Demand (PD)", value=f"{state_row['PD_norm']:.3f}", delta=f"{state_row['contrib_pct_PD']:.1f}% share")
        st.metric(label="Penetration Gap (PG)", value=f"{state_row['PG_norm']:.3f}", delta=f"{state_row['contrib_pct_PG']:.1f}% share")
    with col2:
        st.metric(label="Growth Potential (GP)", value=f"{state_row['GP_norm']:.3f}", delta=f"{state_row['contrib_pct_GP']:.1f}% share")
        st.metric(label="Market Momentum (MMI)", value=f"{state_row['MMI_norm']:.3f}", delta=f"{state_row['contrib_pct_MMI']:.1f}% share")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw Opportunity Score (OPP)", f"{state_row['OPP_score']:.3f}")
    with c2:
        st.metric(label="Logistics Cost Risk Factor", value=f"{state_row['LC_norm']:.3f}", delta=f"-{state_row['risk_penalty_pct']:.1f}% OPP Penalty", delta_color="inverse")

# Render XAI Narrative
def render_xai_narrative(state_row: pd.Series):
    state_code = state_row['customer_state']
    is_mmi_imputed = bool(np.abs(state_row['MMI_norm'] - np.median(df_recalc['MMI_norm'].values)) < 1e-4)
    custom_exp = {
        "state": state_code,
        "rank": int(state_row['EPS_rank']),
        "tier": state_row['tier'],
        "eps_score": float(state_row['EPS_score']),
        "opp_score": float(state_row['OPP_score']),
        "components": {
            "PD": {"norm": float(state_row['PD_norm']), "weight": w_pd_norm, "contrib": float(state_row['contrib_PD']), "contrib_pct": float(state_row['contrib_pct_PD']), "label": "PD"},
            "GP": {"norm": float(state_row['GP_norm']), "weight": w_gp_norm, "contrib": float(state_row['contrib_GP']), "contrib_pct": float(state_row['contrib_pct_GP']), "label": "GP"},
            "PG": {"norm": float(state_row['PG_norm']), "weight": w_pg_norm, "contrib": float(state_row['contrib_PG']), "contrib_pct": float(state_row['contrib_pct_PG']), "label": "PG"},
            "MMI": {"norm": float(state_row['MMI_norm']), "weight": w_mmi_norm, "contrib": float(state_row['contrib_MMI']), "contrib_pct": float(state_row['contrib_pct_MMI']), "label": "MMI"}
        },
        "dominant_driver": state_row['dominant_component'],
        "weakest_component": state_row['weakest_component'],
        "lc_norm": float(state_row['LC_norm']),
        "risk_adj_factor": float(state_row['Risk_Adj']),
        "risk_penalty_abs": float(state_row['risk_penalty_abs']),
        "risk_penalty_pct": float(state_row['risk_penalty_pct']),
        "data_sparse": bool(state_row['data_sparse']),
        "pg_saturated": bool(state_row['PG_norm'] < 0.1),
        "high_lc_flag": bool(state_row['LC_norm'] > 0.7),
        "mmi_imputed": is_mmi_imputed
    }
    
    if is_optimal:
        opt_exp = None
        for exp in xai_report["state_explanations"]:
            if exp["state"] == state_code:
                opt_exp = exp
                break
                
        if opt_exp:
            st.markdown("#### COMPASS-XAI Narrative")
            st.write(opt_exp.get("gemini_narrative_full", opt_exp.get("full_narrative")))
            with st.expander("Show Rule-Based Explanation"):
                st.write(opt_exp["full_narrative"])
        else:
            brief = format_narrative(custom_exp, style='brief')
            full = format_narrative(custom_exp, style='full')
            st.info(brief)
            st.write(full)
    else:
        brief = format_narrative(custom_exp, style='brief')
        full = format_narrative(custom_exp, style='full')
        st.info(brief)
        st.write(full)
        
        st.markdown("#### Dynamic LLM Explanation")
        st.caption("Request COMPASS-XAI to synthesize a narrative for this custom weights scenario.")
        if st.button(f"Generate custom COMPASS-XAI explanation for {state_code}", key=f"gemini_btn_{state_code}", icon=":material/psychology:"):
            with st.spinner("Invoking COMPASS-XAI Engine..."):
                custom_gemini = call_gemini_narrative(custom_exp, xai_report.get('national_stats', {}))
                st.success(custom_gemini.get('full', 'Failed to generate explanation.') if isinstance(custom_gemini, dict) else custom_gemini)


# Load initial data
df_base, w_config, xai_report = load_base_data()
w_star_opt = w_config["w_star"]
gamma_opt = w_config["gamma"]


# ── Title Header ──────────────────────────────────────────────────────────────
st.title("Brazil E-Commerce Expansion Priorities")
st.markdown("Interactive decision support platform using COMPASS-XAI (COMPosite market expansion scoring with Aligned SHAP explanations).")

# ── Settings & Weights (Collapsible Popover) ──────────────────────────────────
with st.popover("⚙️ Adjust Model Weights & Settings: Click to tune parameters and recalculate Priority Scores.", use_container_width=False):
    st.markdown("### Opportunity Component Weights")
    st.caption("Change values to recalculate Priority Scores in real-time.")
    
    # Slide adjusters using state sessions
    if "w_pd" not in st.session_state:
        st.session_state["w_pd"] = float(w_star_opt["PD"])
    if "w_gp" not in st.session_state:
        st.session_state["w_gp"] = float(w_star_opt["GP"])
    if "w_pg" not in st.session_state:
        st.session_state["w_pg"] = float(w_star_opt["PG"])
    if "w_mmi" not in st.session_state:
        st.session_state["w_mmi"] = float(w_star_opt["MMI"])
    if "gamma" not in st.session_state:
        st.session_state["gamma"] = float(gamma_opt)

    w_pd = st.slider("Predicted Demand (PD)", 0.0, 1.0, key="w_pd", step=0.01)
    w_gp = st.slider("Growth Potential (GP)", 0.0, 1.0, key="w_gp", step=0.01)
    w_pg = st.slider("Penetration Gap (PG)", 0.0, 1.0, key="w_pg", step=0.01)
    w_mmi = st.slider("Market Momentum Index (MMI)", 0.0, 1.0, key="w_mmi", step=0.01)
    
    st.markdown("### Logistics Risk Multiplier")
    gamma = st.slider("Risk Penalty (gamma)", 0.0, 1.0, key="gamma", step=0.01)
    
    # Calculate normalization summary
    w_sum = w_pd + w_gp + w_pg + w_mmi
    if w_sum > 0:
        w_pd_norm = w_pd / w_sum
        w_gp_norm = w_gp / w_sum
        w_pg_norm = w_pg / w_sum
        w_mmi_norm = w_mmi / w_sum
    else:
        w_pd_norm = w_gp_norm = w_pg_norm = w_mmi_norm = 0.25
        
    st.markdown("---")
    st.markdown("### Normalized Weights (Sum to 1.0)")
    st.caption(f"- **PD**: {w_pd_norm:.2%} (Optimized: {w_star_opt['PD']:.2%})")
    st.caption(f"- **GP**: {w_gp_norm:.2%} (Optimized: {w_star_opt['GP']:.2%})")
    st.caption(f"- **PG**: {w_pg_norm:.2%} (Optimized: {w_star_opt['PG']:.2%})")
    st.caption(f"- **MMI**: {w_mmi_norm:.2%} (Optimized: {w_star_opt['MMI']:.2%})")
    
    # Reset button
    is_optimal = (
        np.isclose(w_pd, w_star_opt["PD"]) and
        np.isclose(w_gp, w_star_opt["GP"]) and
        np.isclose(w_pg, w_star_opt["PG"]) and
        np.isclose(w_mmi, w_star_opt["MMI"]) and
        np.isclose(gamma, gamma_opt)
    )
    
    if not is_optimal:
        st.info("Weights adjusted. Displaying custom EPS scenario.")
        if st.button("Reset to Optimal Weights", icon=":material/refresh:"):
            st.session_state["w_pd"] = float(w_star_opt["PD"])
            st.session_state["w_gp"] = float(w_star_opt["GP"])
            st.session_state["w_pg"] = float(w_star_opt["PG"])
            st.session_state["w_mmi"] = float(w_star_opt["MMI"])
            st.session_state["gamma"] = float(gamma_opt)
            st.rerun()
    else:
        st.success("Using optimized weights (Max Entropy).")

# Recalculate values in real-time
df_recalc = recalculate_eps(df_base, w_pd_norm, w_gp_norm, w_pg_norm, w_mmi_norm, gamma)

# Precalculate display columns
df_recalc['state_display'] = df_recalc['customer_state'].map(lambda x: f"{STATE_MAP.get(x, x)} ({x})")

# (Title block was moved above Settings)

# ── Main Application Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "National Overview",
    "State Deep-Dive & XAI",
    "Geospatial Maps",
    "Sensitivity & Stability Analysis",
    "PD Model Training"
])

# ── TAB 1: National Overview ──────────────────────────────────────────────────
with tab1:
    st.header("National Performance Overview")
    
    # ── Methodology Banner ──
    with st.container(border=True):
        st.subheader("Expansion Priority Score (EPS) Methodology")
        st.latex(r"EPS = (w_{PD} \times PD + w_{GP} \times GP + w_{PG} \times PG + w_{MMI} \times MMI) \times (1 - \gamma \times LC)")
        st.markdown("**Legend (Click to view detailed formulas):**")
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            with st.popover(f"PD: Predicted Demand (w = {w_pd_norm:.2f})", use_container_width=True):
                st.latex(r"PD = 0.5 \ln(1 + R_{\text{predicted}}) + 0.5 \ln(1 + R_{4w})")
                st.markdown(r"""
                **Meaning:** Evaluates short-term demand scale by blending recent actual revenue with future forecasted revenue (using ARIMA/Prophet models).
                
                **Variables:**
                - $R_{\text{predicted}}$: Forecasted revenue for the upcoming week.
                - $R_{4w}$: Average total revenue over the past 4 weeks.
                - The $\ln(1+x)$ function helps mitigate massive outliers from dominant states.
                """)
            with st.popover(f"GP: Growth Potential (w = {w_gp_norm:.2f})", use_container_width=True):
                st.latex(r"GP = \text{clip}\left(\frac{R_{4w} - R_{8w}}{R_{8w}}, -1, 1\right)")
                st.markdown(r"""
                **Meaning:** Measures short-term revenue growth relative to recent history, indicating market momentum and expansion velocity.
                
                **Variables:**
                - $R_{4w}$: Average revenue over the past 4 weeks.
                - $R_{8w}$: Average revenue over the preceding 8-week cycle.
                - The $\text{clip}(\dots, -1, 1)$ function bounds growth rates between [-100%, +100%] to prevent noise from small denominators.
                """)
        with lcol2:
            with st.popover(f"PG: Penetration Gap (w = {w_pg_norm:.2f})", use_container_width=True):
                st.latex(r"PG = \frac{E[R] - R_{4w}}{E[R]}")
                st.markdown(r"""
                **Meaning:** Assesses the "untapped potential" of a market compared to its theoretical expected baseline.
                
                **Variables:**
                - $E[R]$: Expected revenue, calculated based on State Population and GDP per capita weighting.
                - $R_{4w}$: Actual recent revenue (past 4 weeks).
                - A larger Gap indicates the market has significant room for expansion.
                """)
            with st.popover(f"MMI: Market Momentum Index (w = {w_mmi_norm:.2f})", use_container_width=True):
                st.latex(r"MMI = \ln\left(1 + \frac{R_{4w}}{\text{Sellers}}\right)")
                st.markdown(r"""
                **Meaning:** A market momentum metric reflecting revenue efficiency per active seller. 
                
                **Variables:**
                - $R_{4w}$: Average revenue over the past 4 weeks.
                - $\text{Sellers}$: Count of Active Sellers in the state.
                - A high MMI indicates strong purchasing power but limited supplier availability (low competition).
                """)
        with lcol3:
            with st.popover("LC: Logistics Cost Risk Factor", use_container_width=True):
                st.latex(r"LC_{\text{norm}} = \frac{\text{Freight} - \min}{\max - \min}")
                st.markdown(r"""
                **Meaning:** A risk factor related to transportation costs. High freight costs compress profit margins and limit customer willingness to pay.
                
                **Variables:**
                - $\text{Freight}$: Average freight cost value delivered to the state.
                - Normalized (Min-Max Scaling) to a [0, 1] range (0 = cheapest, 1 = most expensive).
                """)
            with st.popover(f"γ (gamma): Risk Penalty ({gamma:.2f})", use_container_width=True):
                st.latex(r"\text{Risk Adj} = 1 - \gamma \times LC_{\text{norm}}")
                st.markdown(r"""
                **Meaning:** A penalty coefficient that discounts the Opportunity Score if a state exhibits high logistics risk (LC).
                
                **Variables:**
                - $\gamma$ (gamma): Tuning parameter controlling the strictness of the logistics penalty.
                - $LC_{\text{norm}}$: Normalized freight cost score.
                - A lower Risk Adj multiplier heavily reduces the final EPS score.
                """)
    
    # Compute aggregates
    top_row = df_recalc.iloc[0]
    saturated_count = (df_recalc['PG_norm'] < 0.1).sum()
    high_lc_count = (df_recalc['LC_norm'] > 0.7).sum()
    avg_eps = df_recalc['EPS_score'].mean()
    
    # Horizontal KPI Cards Row
    top_state_display = f"{STATE_MAP.get(top_row['customer_state'], top_row['customer_state'])} ({top_row['customer_state']})"
    with st.container(horizontal=True):
        st.metric(
            label="Top Expansion State", 
            value=top_state_display, 
            delta=f"EPS: {top_row['EPS_score']:.1f}", 
            border=True
        )
        st.metric(
            label="National Avg EPS", 
            value=f"{avg_eps:.1f}/100", 
            border=True
        )
        st.metric(
            label="Saturated Markets", 
            value=f"{saturated_count} states", 
            delta="PG < 0.1", 
            delta_color="off",
            border=True
        )
        st.metric(
            label="High Logistics Cost", 
            value=f"{high_lc_count} states", 
            delta="LC > 0.7", 
            delta_color="off",
            border=True
        )

    # ── Final EPS Score Chart ──
    st.subheader("Priority Score (EPS) by State")
    fig_eps = px.bar(
        df_recalc.sort_values("EPS_score", ascending=False),
        x="state_display",
        y="EPS_score",
        color="EPS_score",
        color_continuous_scale="Teal",
        labels={"state_display": "State", "EPS_score": "EPS Score"},
    )
    fig_eps.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2C3E50",
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_eps, use_container_width=True)

    # ── Combo Chart ──
    st.subheader("Component Contributions & Risk Adjustment by State")
    fig_combo = plot_combo_chart(df_recalc, w_pd_norm, w_gp_norm, w_pg_norm, w_mmi_norm)
    st.plotly_chart(fig_combo, use_container_width=True)
    
    # ── Data Table ──
    st.subheader("Expansion Priority rankings")
    df_display = df_recalc[[
        'EPS_rank', 'state_display', 'EPS_score', 'OPP_score', 
        'Risk_Adj', 'dominant_component', 'weakest_component', 'tier', 'data_sparse'
    ]].copy()
    
    # Translate column contents to full descriptions
    df_display['dominant_component'] = df_display['dominant_component'].map(METRIC_NAMES)
    df_display['weakest_component'] = df_display['weakest_component'].map(METRIC_NAMES)
    
    with st.container(border=True):
        st.dataframe(
            df_display,
            column_config={
                "EPS_rank": st.column_config.NumberColumn("Rank", format="%d"),
                "state_display": st.column_config.TextColumn("State"),
                "EPS_score": st.column_config.NumberColumn("EPS Score", format="%.1f"),
                "OPP_score": st.column_config.NumberColumn("Opportunity", format="%.3f"),
                "Risk_Adj": st.column_config.NumberColumn("Risk Adj", format="%.3f"),
                "dominant_component": "Dominant Driver",
                "weakest_component": "Weakest Area",
                "tier": st.column_config.TextColumn("Tier"),
                "data_sparse": "Sparse Data"
            },
            hide_index=True,
            on_select="ignore"
        )

# ── TAB 2: State Deep-Dive & XAI ──────────────────────────────────────────────
with tab2:
    st.header("State-Level Deep-Dive & Comparison")
    
    compare_mode = st.toggle("Enable Comparison Mode", value=False, key="compare_mode")
    state_codes = df_recalc['customer_state'].tolist()
    
    # FIFO Selection Queue Initialization
    if "selection_queue" not in st.session_state:
        st.session_state["selection_queue"] = ["SP"]
        
    queue = st.session_state["selection_queue"]
    
    # Enforce queue lengths and defaults
    if compare_mode:
        if len(queue) < 2:
            default_state_2 = "AM" if queue[0] != "AM" else "SP"
            queue.append(default_state_2)
            st.session_state["selection_queue"] = queue
    else:
        if len(queue) > 1:
            queue = queue[:1]
            st.session_state["selection_queue"] = queue
            
    # Render selectboxes using index mapping (no key bound to prevent Streamlit widget lock issues)
    if compare_mode:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            selected_idx = state_codes.index(queue[0]) if queue[0] in state_codes else 0
            selected_state = st.selectbox(
                "Select Primary State",
                options=state_codes,
                index=selected_idx,
                format_func=lambda x: f"{STATE_MAP.get(x, x)} ({x})"
            )
            if selected_state != queue[0]:
                queue[0] = selected_state
                # Force uniqueness
                if len(queue) > 1 and queue[0] == queue[1]:
                    queue[1] = "AM" if queue[0] != "AM" else "SP"
                st.session_state["selection_queue"] = queue
                st.rerun()
                
        with col_sel2:
            if len(queue) > 1:
                selected_idx_2 = state_codes.index(queue[1]) if queue[1] in state_codes else 1
                selected_state_2 = st.selectbox(
                    "Select Comparison State",
                    options=state_codes,
                    index=selected_idx_2,
                    format_func=lambda x: f"{STATE_MAP.get(x, x)} ({x})"
                )
                if selected_state_2 != queue[1]:
                    queue[1] = selected_state_2
                    # Force uniqueness
                    if queue[0] == queue[1]:
                        queue[0] = "AM" if queue[1] != "AM" else "SP"
                    st.session_state["selection_queue"] = queue
                    st.rerun()
            else:
                # Fallback if queue somehow had 1 element
                st.info("Select a second state from the dropdown or click on the map to compare.")
                selected_state_2 = st.selectbox(
                    "Select Comparison State",
                    options=["None"] + state_codes,
                    index=0,
                    format_func=lambda x: "Choose state..." if x == "None" else f"{STATE_MAP.get(x, x)} ({x})"
                )
                if selected_state_2 != "None":
                    queue.append(selected_state_2)
                    st.session_state["selection_queue"] = queue
                    st.rerun()
    else:
        selected_idx = state_codes.index(queue[0]) if queue[0] in state_codes else 0
        selected_state = st.selectbox(
            "Select a Brazilian state to investigate",
            options=state_codes,
            index=selected_idx,
            format_func=lambda x: f"{STATE_MAP.get(x, x)} ({x})"
        )
        if selected_state != queue[0]:
            st.session_state["selection_queue"] = [selected_state]
            st.rerun()
        selected_state_2 = None

    st.markdown("---")
    
    # ── BENTO BOX LAYOUT ──
    # Prepare rendering queue based on compare_mode
    states_to_render = []
    if compare_mode and len(queue) > 1:
        states_to_render = [(queue[0], "#3b82f6", "Primary"), (queue[1], "#ea580c", "Comparison")]
    else:
        states_to_render = [(queue[0], "#3b82f6", "Selected")]
    
    # ROW 1 (Top Section): Map (Left, 60-65%) and Profiles (Right, 35-40%)
    row1_col1, row1_col2 = st.columns([0.62, 0.38], gap="medium")
    
    with row1_col1:
        with st.container(border=True):
            st.subheader("Interactive Map Selector")
            st.caption("Click directly on any state in the map below to select it. Hover to inspect values.")
            
            # Build the Plotly figure for display
            fig_map_interactive = draw_plotly_brazil_map(df_recalc, queue[0], queue[1] if len(queue) > 1 else None, height=750)
            fig_json = fig_map_interactive.to_json()
            
            # Render component and get selection dictionary
            map_click_data = map_selector(fig_json=fig_json, key="plotly_brazil_map")
            
            # Initialize click timestamp tracking in session state
            if "last_map_click_ts" not in st.session_state:
                st.session_state["last_map_click_ts"] = 0
                
            # Process map clicks securely
            if map_click_data and isinstance(map_click_data, dict):
                click_state = map_click_data.get("state")
                click_ts = map_click_data.get("timestamp", 0)
                
                if click_ts > st.session_state["last_map_click_ts"]:
                    st.session_state["last_map_click_ts"] = click_ts
                    
                    if click_state and click_state in state_codes:
                        if compare_mode:
                            if click_state in queue:
                                if len(queue) > 1:
                                    queue.remove(click_state)
                            else:
                                if len(queue) < 2:
                                    queue.append(click_state)
                                else:
                                    queue = [queue[1], click_state]
                        else:
                            queue = [click_state]
                        
                        st.session_state["selection_queue"] = queue
                        st.rerun()
            
            # Visual selection status indicator
            if compare_mode and len(queue) > 1:
                st.caption(f"🔵 Primary: {STATE_MAP.get(queue[0], queue[0])} ({queue[0]})  ·  🟠 Comparison: {STATE_MAP.get(queue[1], queue[1])} ({queue[1]})")
            else:
                st.caption(f"🔵 Selected: {STATE_MAP.get(queue[0], queue[0])} ({queue[0]})")
                
    with row1_col2:
        if len(states_to_render) > 1:
            tabs_r1 = st.tabs([f"{STATE_MAP.get(s[0], s[0])} ({s[0]})" for s in states_to_render])
        else:
            tabs_r1 = [st.container()]
            
        for idx, (sc, color, label) in enumerate(states_to_render):
            with tabs_r1[idx]:
                state_row = df_recalc[df_recalc['customer_state'] == sc].iloc[0]
                
                with st.container(border=True):
                    render_ranking_card(state_row, color)
                    
                with st.container(border=True):
                    render_radar_chart_section(state_row, color)

    # ── ROW 2 (Bottom Section): Indicator Scores (Left, 40-50%) and XAI Narrative (Right, 50-60%) ──
    row2_col1, row2_col2 = st.columns([0.45, 0.55], gap="medium")
    
    if len(states_to_render) > 1:
        tabs_r2_c1 = row2_col1.tabs([f"{STATE_MAP.get(s[0], s[0])} ({s[0]})" for s in states_to_render])
        tabs_r2_c2 = row2_col2.tabs([f"{STATE_MAP.get(s[0], s[0])} ({s[0]})" for s in states_to_render])
    else:
        tabs_r2_c1 = [row2_col1.container()]
        tabs_r2_c2 = [row2_col2.container()]
        
    for idx, (sc, color, label) in enumerate(states_to_render):
        state_row = df_recalc[df_recalc['customer_state'] == sc].iloc[0]
        
        with tabs_r2_c1[idx]:
            with st.container(border=True):
                st.subheader("Indicator Scores & Weights")
                render_indicator_scores(state_row)
                render_warnings(state_row)
                
        with tabs_r2_c2[idx]:
            with st.container(border=True):
                st.subheader("Explainable AI (XAI) Narrative")
                render_xai_narrative(state_row)

# ── TAB 3: Geospatial Maps ────────────────────────────────────────────────────
with tab3:
    st.header("Geospatial & Component Relationships")
    
    if is_optimal:
        st.subheader("Optimized Spatial Prioritization Map")
        map_path = Config.get_path("reports", "figures_dir") / "fig2_choropleth.png"
        if map_path.exists():
            st.image(str(map_path), caption="Choropleth Maps showing EPS Score, Opportunity Score (OPP), and Logistics Cost (LC)")
        else:
            st.warning("Pre-rendered choropleth map not found.")
    else:
        st.subheader("Recalculated EPS Score Map (Dynamic)")
        st.info("Weights adjusted. Rendering real-time map using GeoPandas...")
        fig = plot_dynamic_map(df_recalc)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("GeoJSON geometry file or geopandas package is not available. Showing pre-rendered map.")
            map_path = Config.get_path("reports", "figures_dir") / "fig2_choropleth.png"
            if map_path.exists():
                st.image(str(map_path), caption="Pre-rendered Choropleth Map (Optimized scenario)")

    st.subheader("Opportunity Component Correlations")
    corr_path = Config.get_path("reports", "figures_dir") / "fig3b_correlation_heatmap.png"
    if corr_path.exists():
        st.image(str(corr_path), caption="Pearson Correlation Heatmap between normalized indicators")

# ── TAB 4: Sensitivity & Stability Analysis ──────────────────────────────────
with tab4:
    st.header("Model Sensitivity & Stability")
    st.markdown("""
    To validate the weights computed by SLSQP Shannon Entropy Maximisation, we run three types of stability analysis:
    - **Monte Carlo Simulations**: Perturbs component values with random Gaussian noise to test ranking robustness.
    - **One-At-A-Time (OAT) Sweeps**: Sequentially sweeps individual weights between [0,1] to detect threshold sensitivity.
    - **Gamma Penalty Sweeps**: Modifies the logistics risk multiplier (gamma) from 0.0 to 1.0 to check ranking shifts.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monte Carlo Simulation")
        mc_path = Config.get_path("reports", "figures_dir") / "fig4_monte_carlo.png"
        if mc_path.exists():
            st.image(str(mc_path), caption="Spearman rank correlation distribution under component noise")
    with col2:
        st.subheader("OAT Weight Sweeps")
        oat_path = Config.get_path("reports", "figures_dir") / "fig5_oat_sweep.png"
        if oat_path.exists():
            st.image(str(oat_path), caption="Rank change profiles for individual weights sweeps")
            
    st.subheader("Logistics Cost Penalty (Gamma) Sweep")
    gamma_path = Config.get_path("reports", "figures_dir") / "fig6_gamma_sweep.png"
    if gamma_path.exists():
        st.image(str(gamma_path), caption="State rankings progression under varying Gamma values")

# ── TAB 5: PD Model Training ──────────────────────────────────────────────────
with tab5:
    st.header("Predicted Demand (PD) Model Training")
    st.markdown("""
    The **Predicted Demand (PD)** formula evaluates short-term demand scale by blending recent actual revenue with future forecasted revenue.
    The forecasting component relies on advanced machine learning models trained on historical data.

    ### 1. Data Preparation & Feature Engineering
    - **Dynamic Features**: Generated state-dependent features such as `revenue_std_4` (rolling standard deviation) and `revenue_momentum`.
    - **Feature Selection**: Highly multicollinear features (e.g., `payment_value`, `unique_customers`) are excluded to prevent data leakage and instability.
    - **Target Transformation**: The target variable `target_next_revenue` is transformed using $y = \log(1+x)$ (`log1p`) to reduce skewness and handle massive outliers from dominant states.

    ### 2. Model Zoo & Walk-Forward Cross Validation
    We evaluate 9 different modeling algorithms using **TimeSeriesSplit (N=5)** Walk-Forward Validation to ensure robust forecasting over time without data leakage:
    1. **Linear Regression (Baseline)**
    2. **Ridge Regression**
    3. **ElasticNet**
    4. **Huber Regressor**
    5. **Random Forest**
    6. **Gradient Boosting**
    7. **XGBoost**
    8. **LightGBM**
    9. **CatBoost**

    **Evaluation Metrics**: RMSE, MAE, WAPE, sMAPE, and MASE. Models are evaluated based on their Skill Score relative to the Baseline Linear Regression.
    """)

    leaderboard_path = Config.get_path("reports", "leaderboard")
    if leaderboard_path.exists():
        st.markdown("#### Model Loss Evaluation & Leaderboard")
        df_lb = pd.read_csv(leaderboard_path)
        # Add arrows to indicate direction of better metrics
        rename_dict = {
            "RMSE": "RMSE (↓)",
            "MAE": "MAE (↓)",
            "WAPE(%)": "WAPE(%) (↓)",
            "sMAPE(%)": "sMAPE(%) (↓)",
            "MASE": "MASE (↓)",
            "SS_RMSE": "SS_RMSE (↑)",
            "Train(s)": "Train(s) (↓)"
        }
        df_lb = df_lb.rename(columns=rename_dict)
        st.dataframe(df_lb, hide_index=True, use_container_width=True)

    cv_path = Config.get_path("reports", "figures_dir") / "cv_metrics_boxplot.png"
    if cv_path.exists():
        st.image(str(cv_path), caption="Cross-Validation Metric Variance (5-Fold Walk-Forward)")

    rp_path = Config.get_path("reports", "figures_dir") / "relative_performance.png"
    if rp_path.exists():
        st.image(str(rp_path), caption="Relative Performance: MASE & Skill Score")

    st.markdown("""
    ### 3. Final Model Selection & SHAP Explanations
    The best-performing model (typically an ensemble method like XGBoost or LightGBM) is selected and evaluated on an unseen test set (e.g., the last 4 weeks of data). 

    Finally, **TreeExplainer SHAP** (SHapley Additive exPlanations) values are extracted from the final model to understand the global and local importance of each feature in predicting the demand, which directly informs the COMPASS-XAI alignment narratives.
    """)

    shap_rf_path = Config.get_path("reports", "figures_dir") / "shap_rf.png"

    if shap_rf_path.exists():
        st.image(str(shap_rf_path), caption="SHAP Feature Importance (Random Forest Explicit)")
