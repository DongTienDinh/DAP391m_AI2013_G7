from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger, short_path
from src.features.transformers import FeatureTransformers

logger = setup_logger("feature_service")


class FeatureEngineeringService:
    """Service for generating and validating weekly feature matrices."""

    def __init__(self, processed_dir: Path, output_file: Path, pred_file: Path):
        self.processed_dir = processed_dir
        self.output_file = output_file
        self.pred_file = pred_file

    def run_feature_pipeline(self, pop_file: Path, gdp_file: Path) -> None:
        logger.info("Starting Feature Engineering Service...")

        # 1. Load all processed data
        raw_tables = self._load_processed_data()
        pop = pd.read_csv(pop_file)
        gdp = pd.read_csv(gdp_file)

        # 2. Build base matrix with all joins
        df = self._build_weekly_base(raw_tables)

        # 3. Re-index to fill empty weeks
        df = self._reindex_fill_gaps(df)

        # 4. Derived features
        df = self._create_derived_features(df)

        # 5. Enrich with external IBGE data
        df = self._add_external_data(df, pop, gdp)

        # 6. Penetration features
        df = self._add_penetration_features(df)

        # 7. Seasonality (sin/cos)
        df = FeatureTransformers.add_seasonality(df)

        # 8. Growth, lags, rolling, and target
        df = self._add_growth_lag_rolling(df)

        # 9. Save
        self._save_results(df)
        logger.info("Feature Engineering Service completed.")

    def _load_processed_data(self) -> dict[str, pd.DataFrame]:
        files = [
            "orders",
            "order_items",
            "customers",
            "sellers",
            "products",
            "order_payments",
            "order_reviews",
        ]
        tables = {}
        for f in files:
            path = self.processed_dir / f"{f}.csv"
            tables[f] = pd.read_csv(path)
        return tables

    def _prepare_orders(self, orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        orders = orders.copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
        orders["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"])
        orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
        orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

        orders["delivery_time_days"] = (
            orders["order_delivered_customer_date"] - orders["order_delivered_carrier_date"]
        ).dt.total_seconds() / 86400.0

        orders["is_late"] = (
            orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
        ).astype(float)

        orders["year_week"] = orders["order_purchase_timestamp"].dt.to_period("W")
        orders["year_int"] = orders["order_purchase_timestamp"].dt.year
        orders["month"] = orders["order_purchase_timestamp"].dt.month

        orders = orders.merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id", how="left"
        )
        return orders

    def _build_weekly_base(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Aggregating transactional data with all joins...")

        orders = self._prepare_orders(tables["orders"], tables["customers"])
        items = tables["order_items"]
        payments = tables["order_payments"]
        reviews = tables["order_reviews"]
        products = tables["products"]
        sellers = tables["sellers"]

        # Aggregate items by order
        items_agg = items.groupby("order_id").agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            item_count=("order_item_id", "count"),
        ).reset_index()
        items_agg["revenue"] = items_agg["total_price"] + items_agg["total_freight"]

        # Aggregate payments by order
        payments_agg = payments.groupby("order_id").agg(
            payment_value=("payment_value", "sum"),
            avg_installments=("payment_installments", "mean"),
        ).reset_index()

        # Aggregate reviews by order
        reviews_agg = reviews.groupby("order_id").agg(
            avg_review_score=("review_score", "mean"),
        ).reset_index()
        reviews_agg["is_positive_sentiment"] = (reviews_agg["avg_review_score"] >= 4).astype(float)

        # Merge all into base
        base = orders.merge(items_agg, on="order_id", how="left")
        base = base.merge(payments_agg, on="order_id", how="left")
        base = base.merge(reviews_agg, on="order_id", how="left")

        # Seller and product info for diversity
        items_seller = items[["order_id", "seller_id", "product_id"]].drop_duplicates()
        items_seller = items_seller.merge(
            sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
        )
        items_seller = items_seller.merge(
            products[["product_id", "product_category_name"]], on="product_id", how="left"
        )

        # Main aggregation at state-week level
        df = (
            base.groupby(["customer_state", "year_week", "year_int", "month"])
            .agg(
                revenue=("revenue", "sum"),
                order_count=("order_id", "nunique"),
                item_count=("item_count", "sum"),
                unique_customers=("customer_unique_id", "nunique"),
                avg_freight_value=("total_freight", "mean"),
                avg_delivery_time=("delivery_time_days", "mean"),
                late_delivery_sum=("is_late", "sum"),
                avg_review_score=("avg_review_score", "mean"),
                positive_reviews=("is_positive_sentiment", "sum"),
                avg_installments=("avg_installments", "mean"),
                payment_value=("payment_value", "sum"),
            )
            .reset_index()
        )

        # Seller and category diversity
        order_info = base[["order_id", "customer_state", "year_week"]].drop_duplicates()
        items_merged = items_seller.merge(order_info, on="order_id", how="left")

        seller_cat_agg = items_merged.groupby(["customer_state", "year_week"]).agg(
            unique_sellers=("seller_id", "nunique"),
            category_diversity=("product_category_name", "nunique"),
        ).reset_index()

        df = df.merge(seller_cat_agg, on=["customer_state", "year_week"], how="left")
        return df

    def _reindex_fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Re-indexing to fill empty weeks...")
        all_states = df["customer_state"].unique()
        all_weeks = pd.period_range(
            start=df["year_week"].min(), end=df["year_week"].max(), freq="W"
        )

        grid = pd.MultiIndex.from_product(
            [all_states, all_weeks], names=["customer_state", "year_week"]
        ).to_frame(index=False)

        df = grid.merge(df, on=["customer_state", "year_week"], how="left")

        fill_0_cols = [
            "revenue", "order_count", "item_count", "unique_customers",
            "late_delivery_sum", "unique_sellers", "category_diversity",
            "positive_reviews", "avg_freight_value", "avg_delivery_time",
            "avg_installments", "payment_value",
        ]
        df[fill_0_cols] = df[fill_0_cols].fillna(0)

        df["year_int"] = df["year_week"].dt.year
        df["month"] = df["year_week"].dt.month

        mask_has_orders = df["order_count"] > 0
        median_score = df.loc[mask_has_orders, "avg_review_score"].median()
        df.loc[mask_has_orders, "avg_review_score"] = df.loc[
            mask_has_orders, "avg_review_score"
        ].fillna(median_score)
        df.loc[df["order_count"] == 0, "avg_review_score"] = 0

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating derived features...")
        df["avg_order_value"] = (df["revenue"] / df["order_count"].replace(0, np.nan)).fillna(0)
        df["late_delivery_rate"] = (df["late_delivery_sum"] / df["order_count"].replace(0, np.nan)).fillna(0)
        df["seller_customer_ratio"] = (df["unique_sellers"] / df["unique_customers"].replace(0, np.nan)).fillna(0)
        df["customer_seller_ratio"] = (df["unique_customers"] / df["unique_sellers"].replace(0, np.nan)).fillna(0)
        df["positive_review_rate"] = (df["positive_reviews"] / df["item_count"].replace(0, np.nan)).fillna(0)
        return df

    def _add_external_data(self, df: pd.DataFrame, pop: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging IBGE metrics...")

        pop_clean = (
            pop.dropna(subset=["state"])
            .rename(columns={"state": "customer_state", "year": "year_int"})
        )
        pop_clean["year_int"] = pop_clean["year_int"].astype(int)
        df["year_int"] = df["year_int"].astype(int)

        df = df.merge(
            pop_clean[["customer_state", "year_int", "population"]],
            on=["customer_state", "year_int"], how="left"
        )

        gdp_agg = (
            gdp[["year", "state", "gdp"]]
            .rename(columns={"state": "customer_state", "year": "year_int"})
        )
        gdp_agg["year_int"] = gdp_agg["year_int"].astype(int)
        df = df.merge(gdp_agg, on=["customer_state", "year_int"], how="left")

        df["gdp_per_capita"] = df["gdp"] / df["population"]
        df["purchasing_power_index"] = df["gdp_per_capita"] / df["gdp_per_capita"].mean()

        ibge_cols = ["population", "gdp", "gdp_per_capita", "purchasing_power_index"]
        df = df.sort_values(["customer_state", "year_week"])
        df[ibge_cols] = df.groupby("customer_state")[ibge_cols].ffill().bfill()

        return df

    def _add_penetration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating penetration features...")
        df["sales_per_capita"] = df["revenue"] / df["population"]
        df["orders_per_capita"] = df["order_count"] / df["population"]
        df["customer_penetration"] = df["unique_customers"] / df["population"]
        df["penetration_gap"] = df["customer_penetration"] - df["customer_penetration"].mean()
        df["seller_density"] = df["unique_sellers"] / df["population"] * 1000
        return df

    def _add_growth_lag_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating growth, lags, rolling features...")
        df = df.sort_values(["customer_state", "year_week"]).reset_index(drop=True)
        grp = df.groupby("customer_state")

        # Growth
        df["revenue_growth_1w"] = grp["revenue"].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)
        df["revenue_growth_4w"] = grp["revenue"].pct_change(4).replace([np.inf, -np.inf], 0).round(4).fillna(0)
        df["order_growth_1w"] = grp["order_count"].pct_change(1).replace([np.inf, -np.inf], 0).round(4).fillna(0)

        # Lags
        df["revenue_lag_1"] = grp["revenue"].shift(1)
        df["revenue_lag_2"] = grp["revenue"].shift(2)
        df["revenue_lag_4"] = grp["revenue"].shift(4)
        df["revenue_lag_8"] = grp["revenue"].shift(8)
        df["orders_lag_1"] = grp["order_count"].shift(1)
        df["customers_lag_1"] = grp["unique_customers"].shift(1)

        df["revenue_lag_4"] = df["revenue_lag_4"].fillna(df["revenue_lag_2"]).fillna(df["revenue_lag_1"])
        df["revenue_lag_8"] = df["revenue_lag_8"].fillna(df["revenue_lag_4"]).fillna(df["revenue_lag_1"])

        # Rolling
        df["revenue_rolling_4"] = grp["revenue"].transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).mean()
        )
        df["revenue_rolling_8"] = grp["revenue"].transform(
            lambda x: x.shift(1).rolling(8, min_periods=4).mean()
        )
        df["revenue_rolling_12"] = grp["revenue"].transform(
            lambda x: x.shift(1).rolling(12, min_periods=6).mean()
        )
        df["revenue_ewm_4"] = grp["revenue"].transform(
            lambda x: x.shift(1).ewm(span=4, min_periods=2).mean()
        )
        df["revenue_ewm_8"] = grp["revenue"].transform(
            lambda x: x.shift(1).ewm(span=8, min_periods=4).mean()
        )

        rolling_cols = [
            "revenue_rolling_4", "revenue_rolling_8", "revenue_rolling_12",
            "revenue_ewm_4", "revenue_ewm_8",
        ]
        for col in rolling_cols:
            df[col] = df[col].fillna(df["revenue_lag_1"])

        # Target
        df["target_next_revenue"] = grp["revenue"].shift(-1)

        return df

    def _save_results(self, df: pd.DataFrame) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save prediction data (rows with lags but no target)
        pred_mask = (
            df["revenue_lag_1"].notna()
            & df["revenue_lag_2"].notna()
            & df["target_next_revenue"].isna()
        )
        pred_df = df[pred_mask]
        pred_df.to_csv(self.pred_file, index=False)
        logger.info(f"Saved {len(pred_df)} prediction rows to {short_path(self.pred_file)}")

        # Training set: drop rows with NaN in core lags or target
        train_df = df.dropna(subset=["revenue_lag_1", "revenue_lag_2", "target_next_revenue"])

        # Filter data from 2017-01-16 onwards (matching original)
        train_df = train_df[train_df["year_week"].dt.start_time >= "2017-01-16"]

        train_df.to_csv(self.output_file, index=False)
        logger.info(f"Saved {len(train_df)} training rows to {short_path(self.output_file)}")
