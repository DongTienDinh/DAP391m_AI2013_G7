import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Tuple, List

from src.olist_pipeline.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

warnings.filterwarnings('ignore')

def load_data(processed_dir: Path, population_file: Path, gdp_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all processed e-commerce tables and external IBGE data.
    """
    logger.info("Loading processed data from CSV...")

    orders = pd.read_csv(processed_dir / "orders.csv", parse_dates=[
        'order_purchase_timestamp',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ])
    items = pd.read_csv(processed_dir / "order_items.csv")
    payments = pd.read_csv(processed_dir / "order_payments.csv")
    reviews = pd.read_csv(processed_dir / "order_reviews.csv")
    products = pd.read_csv(processed_dir / "products.csv")
    customers = pd.read_csv(processed_dir / "customers.csv")
    sellers = pd.read_csv(processed_dir / "sellers.csv")
    population = pd.read_csv(population_file)
    gdp_data = pd.read_csv(gdp_file)

    logger.info(f"   Loaded {len(orders)} orders and {len(population)} IBGE records.")

    return orders, items, payments, reviews, products, customers, sellers, population, gdp_data


def prepare_orders(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches orders with customer state and calculates time-based metrics.
    """
    logger.info("Creating time columns and merging customer info...")

    # Calculate delivery metrics
    orders['delivery_time_days'] = (
        orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']
    ).dt.total_seconds() / 86400.0

    orders['is_late'] = (
        orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
    ).astype(float)

    # Time-series attributes
    orders['year_week'] = orders['order_purchase_timestamp'].dt.to_period('W')
    orders['year_int'] = orders['order_purchase_timestamp'].dt.year
    orders['month'] = orders['order_purchase_timestamp'].dt.month

    # Merge customer geography
    orders = orders.merge(
        customers[['customer_id', 'customer_unique_id', 'customer_state']],
        on='customer_id',
        how='left'
    )

    return orders


def build_base(orders: pd.DataFrame, items: pd.DataFrame, payments: pd.DataFrame, reviews: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates items, payments, and reviews to create an order-level base table.
    """
    logger.info("Merging tables at order level...")

    # Aggregate items
    items_agg = items.groupby('order_id').agg(
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        item_count=('order_item_id', 'count'),
        product_ids=('product_id', list),
        seller_ids=('seller_id', list)
    ).reset_index()

    items_agg['revenue'] = items_agg['total_price'] + items_agg['total_freight']

    # Aggregate payments
    payments_agg = payments.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'mean')
    ).reset_index()

    # Aggregate reviews
    reviews_agg = reviews.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean')
    ).reset_index()
    reviews_agg['is_positive_sentiment'] = (reviews_agg['avg_review_score'] >= 4).astype(float)

    # Merge into base orders table
    base = orders.merge(items_agg, on='order_id', how='left')
    base = base.merge(payments_agg, on='order_id', how='left')
    base = base.merge(reviews_agg, on='order_id', how='left')

    # Prepare helper table for diversity metrics
    items_seller = items[['order_id', 'seller_id', 'product_id']].drop_duplicates()
    items_seller = items_seller.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    items_seller = items_seller.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')

    return base, items_seller


def aggregate_weekly(base: pd.DataFrame, items_seller: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates order data into a state x week panel structure.
    """
    logger.info("Aggregating by state and year_week...")

    # Main metrics aggregation
    df = (base.groupby(['customer_state', 'year_week', 'year_int', 'month'])
          .agg(
              revenue=('revenue', 'sum'),
              order_count=('order_id', 'nunique'),
              item_count=('item_count', 'sum'),
              unique_customers=('customer_unique_id', 'nunique'),
              avg_freight_value=('total_freight', 'mean'),
              avg_delivery_time=('delivery_time_days', 'mean'),
              late_delivery_sum=('is_late', 'sum'),
              avg_review_score=('avg_review_score', 'mean'),
              positive_reviews=('is_positive_sentiment', 'sum'),
              avg_installments=('payment_installments', 'mean'),
              payment_value=('payment_value', 'sum'),
          ).reset_index())

    # Diversity metrics aggregation
    order_info = base[['order_id', 'customer_state', 'year_week']].drop_duplicates()
    items_merged = items_seller.merge(order_info, on='order_id', how='left')

    seller_cat_agg = items_merged.groupby(['customer_state', 'year_week']).agg(
        unique_sellers=('seller_id', 'nunique'),
        category_diversity=('product_category_name', 'nunique'),
    ).reset_index()

    df = df.merge(seller_cat_agg, on=['customer_state', 'year_week'], how='left')

    return df


def reindex_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all state-week combinations exist to prevent gaps in time-series data.
    """
    logger.info("Re-indexing to fill empty weeks (Gap Handling)...")

    all_states = df['customer_state'].unique()
    all_weeks = pd.period_range(start=df['year_week'].min(), end=df['year_week'].max(), freq='W')

    grid = pd.MultiIndex.from_product(
        [all_states, all_weeks],
        names=['customer_state', 'year_week']
    ).to_frame(index=False)

    df = grid.merge(df, on=['customer_state', 'year_week'], how='left')

    # Fill zero for counts and sums
    fill_0_cols = [
        'revenue', 'order_count', 'item_count', 'unique_customers',
        'late_delivery_sum', 'unique_sellers', 'category_diversity',
        'positive_reviews', 'avg_freight_value', 'avg_delivery_time',
        'avg_installments', 'payment_value'
    ]
    df[fill_0_cols] = df[fill_0_cols].fillna(0)

    # Restore date components
    df['year_int'] = df['year_week'].dt.year
    df['month'] = df['year_week'].dt.month

    # Handle scores (impute median for active weeks, 0 for inactive)
    mask_has_orders = df['order_count'] > 0
    median_score = df.loc[mask_has_orders, 'avg_review_score'].median()
    df.loc[mask_has_orders, 'avg_review_score'] = df.loc[mask_has_orders, 'avg_review_score'].fillna(median_score)
    df.loc[df['order_count'] == 0, 'avg_review_score'] = 0

    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates ratios and performance rates.
    """
    logger.info("Creating derived features...")

    df['avg_order_value'] = (df['revenue'] / df['order_count'].replace(0, np.nan)).fillna(0)
    df['late_delivery_rate'] = (df['late_delivery_sum'] / df['order_count'].replace(0, np.nan)).fillna(0)
    df['seller_customer_ratio'] = (df['unique_sellers'] / df['unique_customers'].replace(0, np.nan)).fillna(0)
    df['customer_seller_ratio'] = (df['unique_customers'] / df['unique_sellers'].replace(0, np.nan)).fillna(0)
    df['positive_review_rate'] = (df['positive_reviews'] / df['item_count'].replace(0, np.nan)).fillna(0)

    return df


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cyclical sin/cos encoding to temporal attributes.
    """
    logger.info("Adding seasonality features...")

    week_of_year = df['year_week'].dt.week.astype(float)
    df['week_of_year'] = week_of_year.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
    df['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)

    return df


def add_population_features(df: pd.DataFrame, population: pd.DataFrame, gdp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges external IBGE demographic and economic data.
    """
    logger.info("Adding demographic features from IBGE...")

    # Process Population
    population_clean = (population
                        .dropna(subset=['state'])
                        .rename(columns={'state': 'customer_state', 'year': 'year_int'}))
    population_clean['year_int'] = population_clean['year_int'].astype(int)
    df['year_int'] = df['year_int'].astype(int)

    df = df.merge(population_clean[['customer_state', 'year_int', 'population']], on=['customer_state', 'year_int'], how='left')

    # Process GDP
    gdp_agg = (gdp_data[['year', 'state', 'gdp']]
               .rename(columns={'state': 'customer_state', 'year': 'year_int'}))
    gdp_agg['year_int'] = gdp_agg['year_int'].astype(int)
    df = df.merge(gdp_agg, on=['customer_state', 'year_int'], how='left')

    # Indicators
    df['gdp_per_capita'] = df['gdp'] / df['population']
    df['purchasing_power_index'] = df['gdp_per_capita'] / df['gdp_per_capita'].mean()

    # Fill temporal gaps in IBGE data
    ibge_cols = ['population', 'gdp', 'gdp_per_capita', 'purchasing_power_index']
    df = df.sort_values(['customer_state', 'year_week'])
    df[ibge_cols] = df.groupby('customer_state')[ibge_cols].ffill().bfill()

    return df


def add_penetration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates per-capita penetration and market gap metrics.
    """
    logger.info("Creating penetration features...")

    df['sales_per_capita'] = df['revenue'] / df['population']
    df['orders_per_capita'] = df['order_count'] / df['population']
    df['customer_penetration'] = df['unique_customers'] / df['population']
    df['penetration_gap'] = df['customer_penetration'] - df['customer_penetration'].mean()
    df['seller_density'] = df['unique_sellers'] / df['population'] * 1000

    return df


def add_growth_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates growth rates, lagged revenue, and rolling window statistics.
    """
    logger.info("Calculating growth, lags, and rolling windows...")

    df = df.sort_values(['customer_state', 'year_week']).reset_index(drop=True)
    grp = df.groupby('customer_state')

    # Growth
    df['revenue_growth_1w'] = grp['revenue'].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)
    df['revenue_growth_4w'] = grp['revenue'].pct_change(4).replace([np.inf, -np.inf], 0).round(4).fillna(0)
    df['order_growth_1w'] = grp['order_count'].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)

    # Lags
    df['revenue_lag_1'] = grp['revenue'].shift(1)
    df['revenue_lag_2'] = grp['revenue'].shift(2)
    df['revenue_lag_4'] = grp['revenue'].shift(4)
    df['revenue_lag_8'] = grp['revenue'].shift(8)
    df['orders_lag_1'] = grp['order_count'].shift(1)
    df['customers_lag_1'] = grp['unique_customers'].shift(1)

    # Fill long lags
    df['revenue_lag_4'] = df['revenue_lag_4'].fillna(df['revenue_lag_2']).fillna(df['revenue_lag_1'])
    df['revenue_lag_8'] = df['revenue_lag_8'].fillna(df['revenue_lag_4']).fillna(df['revenue_lag_1'])

    # Rolling Windows (shift(1) prevents leakage)
    df['revenue_rolling_4'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    df['revenue_rolling_8'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean())
    df['revenue_rolling_12'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(12, min_periods=6).mean())

    # EWM
    df['revenue_ewm_4'] = grp['revenue'].transform(lambda x: x.shift(1).ewm(span=4, min_periods=2).mean())
    df['revenue_ewm_8'] = grp['revenue'].transform(lambda x: x.shift(1).ewm(span=8, min_periods=4).mean())

    fill_cols = ['revenue_rolling_4', 'revenue_rolling_8', 'revenue_rolling_12', 'revenue_ewm_4', 'revenue_ewm_8']
    for col in fill_cols:
        df[col] = df[col].fillna(df['revenue_lag_1'])

    # Target
    df['target_next_revenue'] = grp['revenue'].shift(-1)

    return df


def cleanup_and_save(df: pd.DataFrame, output_file: Path, pred_file: Path) -> pd.DataFrame:
    """
    Finalizes the feature set, creates prediction set, and saves to CSV.
    """
    logger.info("Cleaning up and saving files...")

    final_cols = [
        'customer_state', 'year_week', 'revenue', 'order_count', 'item_count',
        'unique_customers', 'avg_freight_value', 'avg_delivery_time',
        'unique_sellers', 'category_diversity', 'avg_review_score',
        'payment_value', 'avg_installments', 'avg_order_value',
        'late_delivery_rate', 'seller_customer_ratio', 'customer_seller_ratio',
        'positive_review_rate', 'week_of_year', 'month_sin', 'month_cos',
        'week_sin', 'week_cos', 'population', 'gdp_per_capita',
        'purchasing_power_index', 'sales_per_capita', 'orders_per_capita',
        'customer_penetration', 'penetration_gap', 'seller_density',
        'revenue_growth_1w', 'revenue_growth_4w', 'order_growth_1w',
        'revenue_lag_1', 'revenue_lag_2', 'revenue_lag_4', 'revenue_lag_8',
        'orders_lag_1', 'customers_lag_1', 'revenue_rolling_4',
        'revenue_rolling_8', 'revenue_rolling_12', 'revenue_ewm_4',
        'revenue_ewm_8', 'target_next_revenue'
    ]
    
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Save future prediction set (target is NaN)
    pred_df = df[df['revenue_lag_1'].notna() & df['revenue_lag_2'].notna() & df['target_next_revenue'].isna()]
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved prediction data (future weeks) to: {pred_file}")

    # Prepare training set
    df = df.dropna(subset=['revenue_lag_1', 'revenue_lag_2', 'target_next_revenue'])
    df = df[df['year_week'].dt.start_time >= '2017-01-16']

    df.to_csv(output_file, index=False)
    logger.info(f"Saved training features to: {output_file}")
    logger.info(f"Final training shape: {df.shape}")

    return df

def run_feature_engineering_pipeline(processed_dir: Path, population_file: Path, gdp_file: Path, output_file: Path, pred_file: Path) -> None:
    """
    Executes the full feature engineering pipeline.
    """
    logger.info("🚀 Starting weekly features generation pipeline...")

    # 1. Load
    orders, items, payments, reviews, products, customers, sellers, population, gdp_data = load_data(processed_dir, population_file, gdp_file)

    # 2. Pipeline stages
    orders = prepare_orders(orders, customers)
    base, items_seller = build_base(orders, items, payments, reviews, products, sellers)
    df = aggregate_weekly(base, items_seller)
    df = reindex_fill_gaps(df)
    df = create_derived_features(df)
    df = add_seasonality(df)
    df = add_population_features(df, population, gdp_data)
    df = add_penetration_features(df)
    df = add_growth_lag_rolling(df)

    # 3. Output
    df = cleanup_and_save(df, output_file, pred_file)

    logger.info("Feature engineering pipeline completed successfully.")
