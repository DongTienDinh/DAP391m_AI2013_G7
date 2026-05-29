# -*- coding: utf-8 -*-
"""
Module: expansion_scoring.py
Objective: Extract, modularize, and professionalize the calculation logic of the Expansion Potential Score (EPS)
          of Brazilian states from Olist data and external data (IBGE population, IBGE GDP).
"""

import os
import sys
import io
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# Set UTF-8 encoding for stdout/stderr to prevent character display errors on Terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Use non-interactive 'Agg' backend for Matplotlib to avoid GUI errors when running as a script
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import geobr
    HAS_GEOBR = True
except Exception:
    HAS_GEOBR = False

warnings.filterwarnings('ignore')

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants for valid Brazilian states
VALID_STATES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
]

# Default configuration
DEFAULT_CONFIG = {
    "rolling_window":            4,
    "use_log_predicted_revenue": True,
    "use_delivered_orders_only": True,
    "growth_clip_lower":         0.01,
    "growth_clip_upper":         0.99,
    "weights": {
        "PD": 0.35,  # Predicted Demand
        "GP": 0.20,  # Growth Potential
        "MS": 0.20,  # Market Size (Market size based on Population & GDP)
        "PG": 0.15,  # Penetration Gap
        "SI": 0.05,  # Seller Index (Seller competition density - penalty)
        "LC": 0.05,  # Logistics Cost (Logistics cost & delivery time - penalty)
    },
    "logistics_mode": "freight_delivery_average",
}


def get_paths(project_root: Path = None) -> dict:
    """
    Dynamically determine and return important project directory and file paths.
    
    Args:
        project_root: Project root path. If None, it will be automatically located from this file's directory.
        
    Returns:
        dict: Contains important Path objects.
    """
    if project_root is None:
        # Path of the current file: src/analysis/expansion_scoring.py -> root is parent of parent of parent (3 levels)
        project_root = Path(__file__).resolve().parents[2]
        
    paths = {
        "root": project_root,
        "processed_olist": project_root / "data" / "processed" / "olist",
        "raw_olist":       project_root / "data" / "raw" / "olist",
        "external":        project_root / "data" / "external",
        "figures_dir":     project_root / "reports" / "figures",
    }
    
    # Create figures directory if it doesn't exist
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    
    # Define specific file paths
    paths["pred_path"]       = paths["processed_olist"] / "predicted_next_week_revenue.csv"
    paths["raw_pop"]         = paths["external"] / "br_ibge_populacao_uf.csv"
    paths["raw_gdp"]         = paths["external"] / "br_ibge_pib_uf.csv"
    paths["br_states_json"]  = paths["external"] / "br_states.geojson"
    paths["raw_geobr"]       = paths["external"] / "geobr_shapefiles"
    
    paths["output_features"] = paths["processed_olist"] / "state_week_eps_features.csv"
    paths["output_ranking"]  = paths["processed_olist"] / "eps_state_ranking.csv"
    
    return paths


def minmax_series(s: pd.Series) -> pd.Series:
    """
    Min-Max normalize a Series to range [0, 1].
    If all values are equal, return a series of zeros.
    """
    s = pd.Series(s).astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mn, mx):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def clip_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Limit Series values by quantiles to remove outlier noise.
    """
    s = pd.Series(s).astype(float)
    return s.clip(s.quantile(lower), s.quantile(upper))


def parse_week_range(s: str) -> tuple:
    """
    Parse a week range string (e.g., '2018-08-27/2018-09-02')
    into start and end Timestamp pair.
    """
    try:
        start, end = str(s).split("/")
        return pd.to_datetime(start), pd.to_datetime(end)
    except Exception as e:
        raise ValueError(f"Cannot parse week range '{s}': {e}")


def attach_nearest_past_year(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    value_cols: list,
    state_col: str = "state",
) -> pd.DataFrame:
    """
    Merge external data (population, GDP) into the panel table based on the nearest past year.
    Helps handle mismatched years between Olist (2016-2018) and IBGE data.
    """
    external = external.sort_values([state_col, "year"])
    rows = []
    
    # Get unique state and year pairs from the panel
    unique_keys = panel[["customer_state", "year"]].drop_duplicates()
    
    for _, row in unique_keys.iterrows():
        st, yr = row["customer_state"], int(row["year"])
        # Find records for this state with a year less than or equal to the current year
        sub = external[(external[state_col] == st) & (external["year"] <= yr)]
        if sub.empty:
            # If no past year exists, get the earliest available year
            sub = external[external[state_col] == st]
            
        record = {"customer_state": st, "year": yr}
        if sub.empty:
            for c in value_cols:
                record[c] = np.nan
        else:
            # Get the record of the nearest largest year in the filtered set
            latest = sub.sort_values("year").iloc[-1]
            for c in value_cols:
                record[c] = latest[c]
        rows.append(record)
        
    rows_df = pd.DataFrame(rows)
    return panel.merge(rows_df, on=["customer_state", "year"], how="left")


def load_table(filename: str, required_cols: list, paths: dict) -> pd.DataFrame:
    """
    Load an Olist table. Prioritize loading from the processed directory; if columns are missing
    or not found, automatically fallback to the raw directory.
    """
    processed_path = paths["processed_olist"] / filename
    raw_path       = paths["raw_olist"] / f"olist_{filename.replace('.csv','')}_dataset.csv"

    for path in [processed_path, raw_path]:
        if path.exists():
            df = pd.read_csv(path)
            if all(c in df.columns for c in required_cols):
                logger.info(f"Successfully loaded '{filename}' from directory '{path.parent.name}'")
                return df

    raise FileNotFoundError(
        f"[ERROR] Table '{filename}' containing columns {required_cols} not found "
        f"in both processed and raw directories."
    )


def load_data(paths: dict) -> dict:
    """
    Load all required input data for the EPS scoring model.
    """
    logger.info("Starting to load input data files...")
    
    # Verify required files exist
    for key in ["pred_path", "raw_pop", "raw_gdp"]:
        p = paths[key]
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] Missing required file: {p}")
            
    # Load next week's predicted revenue
    pred = pd.read_csv(paths["pred_path"])
    required_pred_cols = ["customer_state", "year_week_current", "predicted_next_week_revenue"]
    missing_cols = set(required_pred_cols) - set(pred.columns)
    if missing_cols:
        raise ValueError(f"[ERROR] Predicted revenue file is missing columns: {missing_cols}")
        
    # Load Olist tables
    customers = load_table("customers.csv", ["customer_id", "customer_unique_id", "customer_state"], paths)
    orders    = load_table("orders.csv",    ["order_id", "customer_id", "order_status", "order_purchase_timestamp"], paths)
    items     = load_table("order_items.csv", ["order_id", "seller_id", "price", "freight_value"], paths)
    sellers   = load_table("sellers.csv",   ["seller_id", "seller_state"], paths)
    
    # Load population and GDP data
    population = pd.read_csv(paths["raw_pop"])
    gdp_df     = pd.read_csv(paths["raw_gdp"])
    
    logger.info(f"Successfully loaded all data. Predicted revenue set has {pred.shape[0]} rows.")
    
    return {
        "pred": pred,
        "customers": customers,
        "orders": orders,
        "items": items,
        "sellers": sellers,
        "population": population,
        "gdp": gdp_df
    }


def build_fact_table(orders: pd.DataFrame, customers: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and clean Olist tables to create a detailed transaction Fact table.
    """
    logger.info("Building transaction Fact table...")
    
    # Process time and calculate delivery time in days
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    
    if "order_delivered_customer_date" in orders.columns:
        orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
        orders["delivery_time_days"] = (
            orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400
    else:
        orders["delivery_time_days"] = np.nan

    # Filter for successfully delivered orders only
    if "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"].copy()
        
    # Calculate revenue for each product in the order
    items = items.copy()
    if "revenue" not in items.columns:
        items["revenue"] = items["price"] + items["freight_value"]

    # Merge Fact table
    fact = (
        orders
        .merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")
        .merge(items[["order_id", "seller_id", "price", "freight_value", "revenue"]], on="order_id", how="left")
    )
    
    # Filter valid states and remove rows missing key fields
    fact = fact[fact["customer_state"].isin(VALID_STATES)].copy()
    fact = fact.dropna(subset=["order_purchase_timestamp", "customer_state"])
    
    # Create week period and year columns
    fact["week_period"] = fact["order_purchase_timestamp"].dt.to_period("W-SUN")
    fact["week_start"]  = fact["week_period"].apply(lambda p: p.start_time)
    fact["week_end"]    = fact["week_period"].apply(lambda p: p.end_time.normalize())
    fact["year"]        = fact["week_start"].dt.year
    
    logger.info(f"Fact table completed: {fact.shape[0]:,} transaction rows | "
                f"{fact['customer_state'].nunique()} states | {fact['week_start'].nunique()} transaction weeks.")
    return fact


def aggregate_state_week_features(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Group the detailed Fact table to calculate state-level aggregated features by week.
    Also reindex to create a full time grid (full panel) without missing weeks.
    """
    logger.info("Grouping and aggregating state-level features by week...")
    
    state_week = (
        fact.groupby(["customer_state", "week_start", "week_end", "year"])
        .agg(
            revenue          = ("revenue",            "sum"),
            order_count      = ("order_id",           "nunique"),
            unique_customers = ("customer_unique_id", "nunique"),
            seller_count     = ("seller_id",          "nunique"),
            avg_price        = ("price",              "mean"),
            avg_freight_value= ("freight_value",      "mean"),
            avg_delivery_time= ("delivery_time_days", "mean"),
        )
        .reset_index()
    )

    # Create full week range (Full Grid) from the first week to the last week
    all_weeks = pd.date_range(state_week["week_start"].min(), state_week["week_start"].max(), freq="W-MON")
    panel_idx = pd.MultiIndex.from_product(
        [VALID_STATES, all_weeks], names=["customer_state", "week_start"]
    )
    
    # Reindex the grid
    state_week = (
        state_week.set_index(["customer_state", "week_start"])
        .reindex(panel_idx)
        .reset_index()
    )
    
    state_week["week_end"] = state_week["week_start"] + pd.Timedelta(days=6)
    state_week["year"]     = state_week["week_start"].dt.year
    
    # Fill 0 for count/sum metrics
    for col in ["revenue", "order_count", "unique_customers", "seller_count"]:
        state_week[col] = state_week[col].fillna(0)
        
    # Interpolate (forward-fill / backward-fill) for average metrics (price, freight, delivery time)
    for col in ["avg_price", "avg_freight_value", "avg_delivery_time"]:
        state_week[col] = (
            state_week.groupby("customer_state")[col]
            .transform(lambda s: s.ffill().bfill())
        )
        state_week[col] = state_week[col].fillna(state_week[col].median())
        
    logger.info(f"Successfully built state-week time grid (Full Panel): {state_week.shape[0]} rows.")
    return state_week


def compute_rolling_features(state_week: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate rolling features (historical moving averages) of the state to prevent data leakage.
    """
    logger.info("Calculating historical moving average features (Rolling Features)...")
    
    state_week = state_week.sort_values(["customer_state", "week_start"]).copy()
    grp = state_week.groupby("customer_state")
    W = config["rolling_window"]

    # 1. Shift revenue based on previous weeks (lags)
    for lag in [1, 2, 4, 8]:
        state_week[f"lag_revenue_{lag}w"] = grp["revenue"].shift(lag)

    # 2. Moving average revenue of the last 4 weeks and the preceding 4 weeks to calculate Growth Potential
    state_week["rolling_revenue_4w"] = grp["revenue"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )
    state_week["rolling_revenue_prev_4w"] = grp["revenue"].transform(
        lambda s: s.shift(W + 1).rolling(W, min_periods=1).mean()
    )
    state_week["growth_potential_raw"] = (
        (state_week["rolling_revenue_4w"] - state_week["rolling_revenue_prev_4w"])
        / (state_week["rolling_revenue_prev_4w"].abs() + 1e-6)
    )

    # 3. Rolling number of unique customers over 4 weeks
    state_week["rolling_customers_4w"] = grp["unique_customers"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).sum()
    )
    
    # 4. Rolling average number of sellers over 4 weeks
    state_week["rolling_sellers_4w"] = grp["seller_count"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )

    # 5. Logistics features (freight value and delivery time moving average over 4 weeks)
    state_week["avg_freight_4w"] = grp["avg_freight_value"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )
    state_week["avg_delivery_time_4w"] = grp["avg_delivery_time"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )
    
    return state_week


def calculate_eps(
    pred: pd.DataFrame, 
    state_week: pd.DataFrame, 
    population: pd.DataFrame, 
    gdp_df: pd.DataFrame, 
    config: dict
) -> pd.DataFrame:
    """
    Calculate the composite Expansion Potential Score (EPS) and rank the states.
    """
    logger.info("Merging prediction data and calculating EPS index...")
    
    # Normalize population data
    population = population.copy()
    population["state"] = population["state"].astype(str).str.strip().str.upper()
    population = population[population["state"].isin(VALID_STATES)].copy()
    population = population.dropna(subset=["year", "state", "population"])
    population["year"] = population["year"].astype(int)
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    population = population[population["population"] > 0]
    
    # Normalize GDP data
    gdp_df = gdp_df.copy()
    gdp_df["state"] = gdp_df["state"].astype(str).str.strip().str.upper()
    gdp_df = gdp_df[gdp_df["state"].isin(VALID_STATES)].copy()
    gdp_df = gdp_df.dropna(subset=["year", "state", "gdp"])
    gdp_df["year"] = gdp_df["year"].astype(int)
    gdp_df["gdp"] = pd.to_numeric(gdp_df["gdp"], errors="coerce")
    gdp_df = gdp_df[gdp_df["gdp"] > 0]

    # Merge population and GDP into state_week
    state_week = attach_nearest_past_year(state_week, population, value_cols=["population"])
    state_week = attach_nearest_past_year(state_week, gdp_df, value_cols=["gdp"])
    state_week["gdp_per_capita"] = state_week["gdp"] / state_week["population"]

    # Parse start/end dates from the predicted week
    pred = pred.copy()
    pred[["current_week_start", "current_week_end"]] = pred["year_week_current"].apply(
        lambda x: pd.Series(parse_week_range(x))
    )

    # Merge prediction set with weekly features set
    eps = pred.merge(
        state_week,
        left_on  = ["customer_state", "current_week_start"],
        right_on = ["customer_state", "week_start"],
        how="left"
    )

    # Automatically fix if there is a week period mismatch (e.g., Mon vs Sun)
    if eps["revenue"].isna().all():
        logger.info("[WARN] Week mismatch detected between prediction and panel, realigning periods...")
        pred["current_week_start"] = (
            pred["current_week_start"]
            .dt.to_period("W-MON")
            .apply(lambda p: p.start_time)
        )
        eps = pred.merge(
            state_week,
            left_on  = ["customer_state", "current_week_start"],
            right_on = ["customer_state", "week_start"],
            how="left"
        )

    # Join additional population/GDP if not already present
    if "population" not in eps.columns:
        eps = attach_nearest_past_year(eps, population, ["population"])
    if "gdp" not in eps.columns:
        eps = attach_nearest_past_year(eps, gdp_df, ["gdp"])
        
    eps["gdp_per_capita"] = eps["gdp"] / eps["population"]

    # Check external data integrity
    assert eps.shape[0] == pred.shape[0], "[ERROR] Row counts do not match after merging."
    assert eps["population"].notna().all(), "[ERROR] Some states are missing Population data."
    assert eps["gdp"].notna().all(), "[ERROR] Some states are missing GDP data."

    # ──── CALCULATE EPS COMPONENTS ────
    # 1. PD (Predicted Demand): Expected demand of the next week (using Log1p)
    eps["PD_raw"] = np.log1p(eps["predicted_next_week_revenue"])

    # 2. GP (Growth Potential): Rolling revenue growth potential
    eps["GP_raw"] = clip_series(
        eps["growth_potential_raw"],
        lower=config["growth_clip_lower"],
        upper=config["growth_clip_upper"],
    )

    # 3. MS (Market Size): Market size (Population combined with GDP per capita)
    national_avg_gdppc = eps["gdp"].sum() / eps["population"].sum()
    eps["gdp_per_capita_index"] = eps["gdp_per_capita"] / national_avg_gdppc
    eps["MS_raw"] = np.log1p(eps["population"]) * eps["gdp_per_capita_index"]

    # 4. PG (Penetration Gap): Penetration gap (Market size minus current customer rate)
    eps["MS_norm_temp"]          = minmax_series(eps["MS_raw"])
    eps["penetration_norm_temp"] = minmax_series(
        eps["rolling_customers_4w"] / eps["population"]
    )
    eps["PG_raw"] = eps["MS_norm_temp"] - eps["penetration_norm_temp"]

    # 5. SI (Seller Index): Seller competition density (number of sellers per 100k population)
    eps["SI_raw"] = eps["rolling_sellers_4w"] / eps["population"] * 100_000

    # 6. LC (Logistics Cost): Logistics index (combining freight value and delivery time)
    if config["logistics_mode"] == "freight_delivery_average":
        eps["LC_raw"] = (
            0.5 * minmax_series(eps["avg_freight_4w"])
            + 0.5 * minmax_series(eps["avg_delivery_time_4w"])
        )
    else:
        eps["LC_raw"] = minmax_series(eps["avg_freight_4w"])

    # Min-Max normalize all components to range [0, 1]
    for raw_col in ["PD_raw", "GP_raw", "MS_raw", "PG_raw", "SI_raw", "LC_raw"]:
        norm_col = raw_col.replace("_raw", "_norm")
        eps[norm_col] = minmax_series(eps[raw_col])

    # Calculate weighted composite EPS score
    W = config["weights"]
    eps["EPS"] = (
         W["PD"] * eps["PD_norm"]
      +  W["GP"] * eps["GP_norm"]
      +  W["MS"] * eps["MS_norm"]
      +  W["PG"] * eps["PG_norm"]
      -  W["SI"] * eps["SI_norm"]
      -  W["LC"] * eps["LC_norm"]
    )

    # Normalize EPS score to [0, 100] scale
    eps["EPS_0_100"] = 100 * (
        (eps["EPS"] - eps["EPS"].min())
        / (eps["EPS"].max() - eps["EPS"].min() + 1e-9)
    )
    
    # Rank states based on EPS score
    eps["rank_eps"] = eps["EPS_0_100"].rank(ascending=False, method="dense").astype(int)
    
    # Create interpretation column
    eps["interpretation"] = eps.apply(interpret_eps, axis=1)
    
    logger.info("Completed score calculation and EPS ranking.")
    return eps


def interpret_eps(row) -> str:
    """
    Analyze components to explain why a state gets a high/low EPS score.
    """
    reasons = []
    if row["PD_norm"] >= 0.75: reasons.append("high predicted demand")
    if row["GP_norm"] >= 0.75: reasons.append("strong growth")
    if row["MS_norm"] >= 0.75: reasons.append("large market size")
    if row["PG_norm"] >= 0.75: reasons.append("substantial penetration room")
    if row["SI_norm"] >= 0.75: reasons.append("high seller competition density")
    if row["LC_norm"] >= 0.75: reasons.append("high logistics cost")
    
    return ", ".join(reasons) if reasons else "balanced development profile"


def save_outputs(eps_df: pd.DataFrame, paths: dict) -> None:
    """
    Save output CSV result files.
    """
    # 1. Save the full EPS features dataset
    eps_df.to_csv(paths["output_features"], index=False)
    logger.info(f"Saved EPS features file: {paths['output_features']}")

    # 2. Filter necessary columns for the ranking table and save
    ranking_cols = [
        "customer_state", "year_week_current", "predicted_next_week_revenue",
        "PD_norm", "GP_norm", "MS_norm", "PG_norm", "SI_norm", "LC_norm",
        "EPS", "EPS_0_100", "rank_eps", "interpretation"
    ]
    eps_ranking = eps_df[ranking_cols].sort_values("rank_eps").reset_index(drop=True)
    eps_ranking.to_csv(paths["output_ranking"], index=False)
    logger.info(f"Saved expansion potential ranking table: {paths['output_ranking']}")


def plot_visualizations(eps_ranking: pd.DataFrame, paths: dict) -> None:
    """
    Plot and save result visualization charts:
    1. Horizontal bar chart of the Top 15 states with the greatest expansion potential.
    2. Choropleth Heatmap of opportunity distribution across Brazil.
    """
    logger.info("Creating visual report charts...")
    
    # 1. Plot horizontal bar chart
    top_states = eps_ranking.sort_values("EPS_0_100", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top_states["customer_state"][::-1],
        top_states["EPS_0_100"][::-1],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_states))),
        edgecolor="white",
    )
    ax.set_xlabel("Expansion Potential EPS Score (0–100)", fontsize=10)
    ax.set_ylabel("State", fontsize=10)
    ax.set_title("Top 15 Brazilian states with the largest market expansion potential (EPS)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    
    # Label scores next to each bar
    for bar, v in zip(bars, top_states["EPS_0_100"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=8, fontweight="bold")
                
    plt.tight_layout()
    bar_chart_path = paths["figures_dir"] / "eps_state_ranking_bar.png"
    plt.savefig(bar_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved horizontal bar chart at: {bar_chart_path}")

    # 2. Plot Choropleth heatmap
    states_geo = None
    
    # Prioritize loading from local GeoJSON file
    if paths["br_states_json"].exists():
        try:
            states_geo = gpd.read_file(paths["br_states_json"])
            logger.info("Successfully loaded Brazil map from local GeoJSON file.")
        except Exception as e:
            logger.warning(f"Error reading local GeoJSON file: {e}. Falling back to geobr...")
            
    # Fallback to geobr library
    if states_geo is None and HAS_GEOBR:
        try:
            states_geo = geobr.read_state(year=2018)
            logger.info("Successfully loaded Brazil map from online geobr library.")
        except Exception as e:
            logger.warning(f"Error loading map using geobr: {e}. Falling back to local shapefile...")

    # Final fallback to local shapefiles
    if states_geo is None:
        shp_files = list(paths["raw_geobr"].rglob("*.shp"))
        if not shp_files:
            logger.error("No Brazil map found to plot heatmap.")
            return
        
        # Prioritize state boundary file (states)
        states_shp = [s for s in shp_files if "states" in s.name]
        selected_shp = states_shp[0] if states_shp else shp_files[0]
        try:
            states_geo = gpd.read_file(selected_shp)
            logger.info(f"Successfully loaded map from local shapefile: {selected_shp.name}")
        except Exception as e:
            logger.error(f"Cannot read shapefile: {e}")
            return

    # Normalize map state column name to 'customer_state' for merging
    state_col_candidates = ["abbrev_state", "sigla_uf", "SIGLA_UF", "UF", "uf"]
    matched = [c for c in state_col_candidates if c in states_geo.columns]
    if not matched:
        logger.error(f"State abbreviation column not found. Available columns: {states_geo.columns.tolist()}")
        return
        
    states_geo = states_geo.rename(columns={matched[0]: "customer_state"})
    
    # Merge map with EPS ranking
    gdf = states_geo.merge(eps_ranking[["customer_state", "EPS_0_100", "rank_eps"]],
                           on="customer_state", how="left")

    # Plot and export map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column="EPS_0_100", ax=ax, legend=True, cmap="YlOrRd",
        edgecolor="black", linewidth=0.4,
        missing_kwds={"color": "lightgrey", "label": "No data"},
        legend_kwds={'label': "Expansion Potential Score (EPS)", 'orientation': "horizontal"}
    )
    ax.set_title("Expansion Potential Score (EPS) Map — Brazil States", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    
    heatmap_path = paths["figures_dir"] / "eps_state_heatmap.png"
    plt.savefig(heatmap_path, dpi=250, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap at: {heatmap_path}")


def run_scoring_pipeline(config_override: dict = None, project_root: Path = None) -> pd.DataFrame:
    """
    Run the entire Expansion Potential Score (EPS) pipeline from end to end.
    
    Args:
        config_override: Dictionary to override default config (e.g., weights or rolling window).
        project_root: Project root directory.
        
    Returns:
        pd.DataFrame: Complete EPS data table of states.
    """
    logger.info("=" * 60)
    logger.info("STARTING EXPANSION POTENTIAL SCORE (EPS) PIPELINE")
    logger.info("=" * 60)
    
    # Apply configuration and update if override exists
    config = dict(DEFAULT_CONFIG)
    if config_override:
        if "weights" in config_override and isinstance(config_override["weights"], dict):
            config["weights"].update(config_override["weights"])
        # Update other configuration parameters
        for k, v in config_override.items():
            if k != "weights":
                config[k] = v
                
    logger.info(f"Applied configuration: {config}")

    # Determine file paths
    paths = get_paths(project_root)
    
    # 1. Load input data
    data = load_data(paths)
    
    # 2. Build transaction Fact table
    fact = build_fact_table(data["orders"], data["customers"], data["items"])
    
    # 3. Group state-level features by week
    state_week = aggregate_state_week_features(fact)
    
    # 4. Calculate rolling features
    state_week = compute_rolling_features(state_week, config)
    
    # 5. Calculate composite EPS score
    eps_df = calculate_eps(data["pred"], state_week, data["population"], data["gdp"], config)
    
    # 6. Save results to CSV file
    save_outputs(eps_df, paths)
    
    # 7. Visualize results
    ranking_cols = [
        "customer_state", "year_week_current", "predicted_next_week_revenue",
        "PD_norm", "GP_norm", "MS_norm", "PG_norm", "SI_norm", "LC_norm",
        "EPS", "EPS_0_100", "rank_eps", "interpretation"
    ]
    eps_ranking = eps_df[ranking_cols].sort_values("rank_eps").reset_index(drop=True)
    plot_visualizations(eps_ranking, paths)
    
    logger.info("=" * 60)
    logger.info("EPS PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return eps_df


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run the state expansion potential scoring (EPS) pipeline")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--rolling-window", type=int, default=4, help="Rolling window (number of weeks)")
    parser.add_argument("--w-pd", type=float, default=0.35, help="Weight for PD (Predicted Demand)")
    parser.add_argument("--w-gp", type=float, default=0.20, help="Weight for GP (Growth Potential)")
    parser.add_argument("--w-ms", type=float, default=0.20, help="Weight for MS (Market Size)")
    parser.add_argument("--w-pg", type=float, default=0.15, help="Weight for PG (Penetration Gap)")
    parser.add_argument("--w-si", type=float, default=0.05, help="Weight for SI (Seller Index)")
    parser.add_argument("--w-lc", type=float, default=0.05, help="Weight for LC (Logistics Cost)")
    
    args = parser.parse_args()
    
    # Build override config from CLI
    config_override = {
        "rolling_window": args.rolling_window,
        "weights": {
            "PD": args.w_pd,
            "GP": args.w_gp,
            "MS": args.w_ms,
            "PG": args.w_pg,
            "SI": args.w_si,
            "LC": args.w_lc,
        }
    }
    
    root_path = Path(args.project_root) if args.project_root else None
    
    try:
        run_scoring_pipeline(config_override=config_override, project_root=root_path)
    except Exception as e:
        logger.exception(f"[FATAL ERROR] Unrecoverable error running pipeline: {e}")
        sys.exit(1)
