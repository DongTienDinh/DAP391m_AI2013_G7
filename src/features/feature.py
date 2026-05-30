import sys
import io
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path
import warnings

# Force UTF-8 stdout and stderr encoding for safe terminal logs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "olist"
POPULATION_FILE = PROJECT_ROOT / "data" / "external" / "br_ibge_populacao_uf.csv"
GDP_FILE = PROJECT_ROOT / "data" / "external" / "br_ibge_pib_uf.csv"
OUTPUT_FILE_PROCESSED = PROCESSED_DIR / "features_weekly.csv"
OUTPUT_FILE_NOTEBOOKS = PROJECT_ROOT / "notebooks" / "features_weekly.csv"


def load_data():
    """Load all processed CSV files."""
    print("1. Loading processed data from CSV...")

    orders = pd.read_csv(PROCESSED_DIR / "orders.csv", parse_dates=[
        'order_purchase_timestamp',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ])
    items = pd.read_csv(PROCESSED_DIR / "order_items.csv")
    payments = pd.read_csv(PROCESSED_DIR / "order_payments.csv")
    reviews = pd.read_csv(PROCESSED_DIR / "order_reviews.csv")
    products = pd.read_csv(PROCESSED_DIR / "products.csv")
    customers = pd.read_csv(PROCESSED_DIR / "customers.csv")
    sellers = pd.read_csv(PROCESSED_DIR / "sellers.csv")

    # Load IBGE population data
    population = pd.read_csv(POPULATION_FILE)

    # Load IBGE GDP data
    gdp_data = pd.read_csv(GDP_FILE)

    print(f"   Orders:    {orders.shape}")
    print(f"   Items:     {items.shape}")
    print(f"   Payments:  {payments.shape}")
    print(f"   Reviews:   {reviews.shape}")
    print(f"   Products:  {products.shape}")
    print(f"   Customers: {customers.shape}")
    print(f"   Sellers:   {sellers.shape}")
    print(f"   IBGE Pop:  {population.shape}")
    print(f"   IBGE GDP:  {gdp_data.shape}")

    return orders, items, payments, reviews, products, customers, sellers, population, gdp_data


def prepare_orders(orders, customers):
    """Create time columns and merge customer_state."""
    print("2. Creating time columns...")

    # Calculate delivery_time_days (since this column is not available in CSV)
    orders['delivery_time_days'] = (
        orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']
    ).dt.total_seconds() / 86400.0

    # Calculate is_late
    orders['is_late'] = (
        orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
    ).astype(float)

    # Create year_week, year_int, month
    orders['year_week'] = orders['order_purchase_timestamp'].dt.to_period('W')
    orders['year_int'] = orders['order_purchase_timestamp'].dt.year
    orders['month'] = orders['order_purchase_timestamp'].dt.month

    # Merge customer_state & customer_unique_id
    orders = orders.merge(
        customers[['customer_id', 'customer_unique_id', 'customer_state']],
        on='customer_id',
        how='left'
    )

    return orders


def build_base(orders, items, payments, reviews, products, sellers):
    """Merge all tables into a base table."""
    print("3. Merging tables (Merge)...")

    # --- Pre-aggregate items by order_id ---
    items_agg = items.groupby('order_id').agg(
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        item_count=('order_item_id', 'count'),
        product_ids=('product_id', list),
        seller_ids=('seller_id', list)
    ).reset_index()

    # Revenue = price + freight
    items_agg['revenue'] = items_agg['total_price'] + items_agg['total_freight']

    # --- Pre-aggregate payments theo order_id ---
    payments_agg = payments.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'mean')
    ).reset_index()

    # --- Pre-aggregate reviews theo order_id ---
    reviews_agg = reviews.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean')
    ).reset_index()
    reviews_agg['is_positive_sentiment'] = (reviews_agg['avg_review_score'] >= 4).astype(float)

    # --- Merge into orders ---
    base = orders.merge(items_agg, on='order_id', how='left')
    base = base.merge(payments_agg, on='order_id', how='left')
    base = base.merge(reviews_agg, on='order_id', how='left')

    # --- Explode to get individual product_id for category_diversity ---
    # Note: Do not explode entirely, only unique product_id + seller_id per order
    # Keep base as is, but need seller_id and product_category to aggregate

    # Merge seller_state via items (get first seller or all)
    items_seller = items[['order_id', 'seller_id', 'product_id']].drop_duplicates()
    items_seller = items_seller.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    items_seller = items_seller.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')

    return base, items_seller


def aggregate_weekly(base, items_seller):
    """Aggregate by state + year_week."""
    print("4. Aggregating by state and year_week...")

    # --- Main aggregation from base (order-level) ---
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

    # --- Aggregate seller & category diversity from items_seller ---
    # Merge order-level info (customer_state, year_week) into items_seller
    order_info = base[['order_id', 'customer_state', 'year_week']].drop_duplicates()
    items_merged = items_seller.merge(order_info, on='order_id', how='left')

    seller_cat_agg = items_merged.groupby(['customer_state', 'year_week']).agg(
        unique_sellers=('seller_id', 'nunique'),
        category_diversity=('product_category_name', 'nunique'),
    ).reset_index()

    df = df.merge(seller_cat_agg, on=['customer_state', 'year_week'], how='left')

    return df


def reindex_fill_gaps(df):
    """Re-indexing to fill empty weeks (Gap Problem)."""
    print("   -> Performing Re-indexing to fill empty weeks...")

    all_states = df['customer_state'].unique()
    all_weeks = pd.period_range(start=df['year_week'].min(), end=df['year_week'].max(), freq='W')

    grid = pd.MultiIndex.from_product(
        [all_states, all_weeks],
        names=['customer_state', 'year_week']
    ).to_frame(index=False)

    df = grid.merge(df, on=['customer_state', 'year_week'], how='left')

    # Fill 0 for count/sum columns when week is empty = no orders
    fill_0_cols = [
        'revenue', 'order_count', 'item_count', 'unique_customers',
        'late_delivery_sum', 'unique_sellers', 'category_diversity',
        'positive_reviews', 'avg_freight_value', 'avg_delivery_time',
        'avg_installments', 'payment_value'
    ]
    df[fill_0_cols] = df[fill_0_cols].fillna(0)

    # Restore year_int and month
    df['year_int'] = df['year_week'].dt.year
    df['month'] = df['year_week'].dt.month

    # Process avg_review_score
    mask_has_orders = df['order_count'] > 0
    median_score = df.loc[mask_has_orders, 'avg_review_score'].median()
    df.loc[mask_has_orders, 'avg_review_score'] = df.loc[mask_has_orders, 'avg_review_score'].fillna(median_score)
    df.loc[df['order_count'] == 0, 'avg_review_score'] = 0

    return df


def create_derived_features(df):
    """Create derived features."""
    print("5. Creating derived features...")

    df['avg_order_value'] = (df['revenue'] / df['order_count'].replace(0, np.nan)).fillna(0)
    df['late_delivery_rate'] = (df['late_delivery_sum'] / df['order_count'].replace(0, np.nan)).fillna(0)
    df['seller_customer_ratio'] = (df['unique_sellers'] / df['unique_customers'].replace(0, np.nan)).fillna(0)
    df['customer_seller_ratio'] = (df['unique_customers'] / df['unique_sellers'].replace(0, np.nan)).fillna(0)
    df['positive_review_rate'] = (df['positive_reviews'] / df['item_count'].replace(0, np.nan)).fillna(0)

    return df


def add_seasonality(df):
    """Add Seasonality features (sin/cos encoding)."""
    print("6. Adding Seasonality features...")

    week_of_year = df['year_week'].dt.week.astype(float)
    df['week_of_year'] = week_of_year.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
    df['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)

    return df


def add_population_features(df, population, gdp_data):
    """Add Demographic features from real IBGE Population CSV and GDP CSV."""
    print("7. Adding Demographic features from IBGE Population + GDP...")

    # Population
    population_clean = (population
                        .dropna(subset=['state'])
                        .rename(columns={'state': 'customer_state', 'year': 'year_int'}))
    population_clean['year_int'] = population_clean['year_int'].astype(int)
    df['year_int'] = df['year_int'].astype(int)

    df = df.merge(population_clean[['customer_state', 'year_int', 'population']], on=['customer_state', 'year_int'], how='left')

    # GDP
    gdp_agg = (gdp_data[['year', 'state', 'gdp']]
               .rename(columns={'state': 'customer_state', 'year': 'year_int'}))
    gdp_agg['year_int'] = gdp_agg['year_int'].astype(int)
    df = df.merge(gdp_agg, on=['customer_state', 'year_int'], how='left')

    # Calculate gdp_per_capita and purchasing_power_index
    df['gdp_per_capita'] = df['gdp'] / df['population']
    df['purchasing_power_index'] = df['gdp_per_capita'] / df['gdp_per_capita'].mean()

    # Forward fill / Back fill for years without IBGE data
    ibge_cols = ['population', 'gdp', 'gdp_per_capita', 'purchasing_power_index']
    df = df.sort_values(['customer_state', 'year_week'])
    df[ibge_cols] = df.groupby('customer_state')[ibge_cols].ffill().bfill()

    return df


def add_penetration_features(df):
    """Create Penetration features (per capita) from population."""
    print("8. Creating Penetration features...")

    df['sales_per_capita'] = df['revenue'] / df['population']
    df['orders_per_capita'] = df['order_count'] / df['population']
    df['customer_penetration'] = df['unique_customers'] / df['population']
    df['penetration_gap'] = df['customer_penetration'] - df['customer_penetration'].mean()
    df['seller_density'] = df['unique_sellers'] / df['population'] * 1000

    return df


def add_growth_lag_rolling(df):
    """Calculate Growth, Lag, Rolling, EWM features."""
    print("9. Calculating Growth, Lag, Rolling...")

    df = df.sort_values(['customer_state', 'year_week']).reset_index(drop=True)
    grp = df.groupby('customer_state')

    # --- Growth (replace inf with 0) ---
    df['revenue_growth_1w'] = grp['revenue'].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)
    df['revenue_growth_4w'] = grp['revenue'].pct_change(4).replace([np.inf, -np.inf], 0).round(4).fillna(0)
    df['order_growth_1w'] = grp['order_count'].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)

    # --- Lag ---
    df['revenue_lag_1'] = grp['revenue'].shift(1)
    df['revenue_lag_2'] = grp['revenue'].shift(2)
    df['revenue_lag_4'] = grp['revenue'].shift(4)
    df['revenue_lag_8'] = grp['revenue'].shift(8)
    df['orders_lag_1'] = grp['order_count'].shift(1)
    df['customers_lag_1'] = grp['unique_customers'].shift(1)

    # Fill null reasonably for long lags (fallback to closer lags)
    df['revenue_lag_4'] = df['revenue_lag_4'].fillna(df['revenue_lag_2']).fillna(df['revenue_lag_1'])
    df['revenue_lag_8'] = df['revenue_lag_8'].fillna(df['revenue_lag_4']).fillna(df['revenue_lag_1'])

    # --- Rolling / EWM (shift(1) to avoid data leakage) ---
    df['revenue_rolling_4'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    df['revenue_rolling_8'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean())
    df['revenue_rolling_12'] = grp['revenue'].transform(lambda x: x.shift(1).rolling(12, min_periods=6).mean())

    df['revenue_ewm_4'] = grp['revenue'].transform(lambda x: x.shift(1).ewm(span=4, min_periods=2).mean())
    df['revenue_ewm_8'] = grp['revenue'].transform(lambda x: x.shift(1).ewm(span=8, min_periods=4).mean())

    for col in ['revenue_rolling_4', 'revenue_rolling_8', 'revenue_rolling_12', 'revenue_ewm_4', 'revenue_ewm_8']:
        df[col] = df[col].fillna(df['revenue_lag_1'])

    # --- Target ---
    df['target_next_revenue'] = grp['revenue'].shift(-1)

    return df


def cleanup_and_save(df):
    """Clean up and save a single features_weekly.csv file (including target_next_revenue)."""
    print("10. Cleaning up and saving file...")

    # Drop rows with NaNs in core lags or target
    df = df.dropna(subset=['revenue_lag_1', 'revenue_lag_2', 'target_next_revenue'])

    # Lọc lấy dữ liệu từ ngày 16/1/2017 trở đi
    df = df[df['year_week'].dt.start_time >= '2017-01-16']

    # Select exactly the 46 requested columns in order
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
    df = df[final_cols]

    # Save all to result files (including target_next_revenue)
    df.to_csv(OUTPUT_FILE_PROCESSED, index=False)
    df.to_csv(OUTPUT_FILE_NOTEBOOKS, index=False)
    print(f"\n✅ Saved file: {OUTPUT_FILE_PROCESSED}")
    print(f"✅ Saved file: {OUTPUT_FILE_NOTEBOOKS}")
    print(f"📊 Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Null statistics
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) > 0:
        print("\n⚠️ Remaining Null columns (includes initial lags & final target - this is normal):")
        print(null_cols)
    else:
        print("\n✅ No Null data remaining in the pipeline.")

    return df


def main():
    print("🚀 Starting weekly features generation (from CSV + IBGE Population)...\n")

    # 1. Load data
    orders, items, payments, reviews, products, customers, sellers, population, gdp_data = load_data()

    # 2. Prepare orders (thêm time cols, delivery_time_days, is_late, customer info)
    orders = prepare_orders(orders, customers)

    # 3. Build base table (merge orders + items + payments + reviews)
    base, items_seller = build_base(orders, items, payments, reviews, products, sellers)

    # 4. Aggregate theo state + year_week
    df = aggregate_weekly(base, items_seller)

    # 5. Re-index để điền gap
    df = reindex_fill_gaps(df)

    # 6. Features phái sinh (avg_order_value, late_delivery_rate, ...)
    df = create_derived_features(df)

    # 7. Seasonality (sin/cos encoding)
    df = add_seasonality(df)

    # 8. Demographic features từ IBGE Population + GDP
    df = add_population_features(df, population, gdp_data)

    # 9. Penetration features (per capita)
    df = add_penetration_features(df)

    # 10. Growth, Lag, Rolling, EWM, Target
    df = add_growth_lag_rolling(df)

    # 11. Cleanup & Save
    df = cleanup_and_save(df)

    # --- Print features summary ---
    print("\n" + "=" * 60)
    print("📋 FINAL FEATURES LIST:")
    print("=" * 60)
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    print("=" * 60)


if __name__ == '__main__':
    main()
