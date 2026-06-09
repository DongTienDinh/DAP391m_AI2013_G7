import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.olist_pipeline.core.logging_setup import setup_logger
from src.olist_pipeline.core.exceptions import DataValidationError
from src.olist_pipeline.data.loader import RawDataLoader
from src.olist_pipeline.data.schema import OLIST_SCHEMAS

logger = setup_logger("data_service")

class DataService:
    """Service for cleaning and preparing Olist data."""

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.loader = RawDataLoader(raw_dir)
        self.processed_dir = processed_dir

    def run_cleaning_pipeline(self) -> None:
        """Executes the full cleaning workflow."""
        logger.info("Starting Data Cleaning Service...")
        
        # 1. Ingest
        self.loader.download_if_missing()
        datasets = self.loader.load_all()
        
        # 2. Validate
        self._validate_raw_data(datasets)
        
        # 3. Clean
        cleaned = self._transform(datasets)
        
        # 4. Save
        self._save(cleaned)
        logger.info("Data Cleaning Service completed.")

    def _validate_raw_data(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Checks for minimum required columns and data presence."""
        for key, schema in OLIST_SCHEMAS.items():
            df = datasets.get(key)
            if df is None:
                raise DataValidationError(f"Required dataset '{key}' missing from loaded data.")
            
            missing_cols = [c for c in schema.required_columns if c not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Dataset '{key}' missing columns: {missing_cols}")
            
            if len(df) < schema.min_rows:
                raise DataValidationError(f"Dataset '{key}' has only {len(df)} rows (min expected: {schema.min_rows})")

    def _transform(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Performs domain-specific cleaning transformations."""
        logger.info("Transforming datasets...")
        
        # Orders cleaning
        orders = datasets['orders']
        orders = orders[orders['order_status'] == 'delivered'].copy()
        orders = orders.dropna(subset=['order_delivered_carrier_date', 'order_delivered_customer_date'])
        
        # Products cleaning
        products = datasets['products']
        products = products.dropna(subset=['product_category_name', 'product_weight_g']).copy()
        
        # Reviews cleaning (drop text columns)
        reviews = datasets['order_reviews'].drop(columns=['review_comment_title', 'review_comment_message'], errors='ignore')
        
        return {
            'orders': orders,
            'products': products,
            'order_reviews': reviews,
            'customers': datasets['customers'],
            'order_items': datasets['order_items'],
            'order_payments': datasets['order_payments'],
            'sellers': datasets['sellers'],
            'geolocation': datasets['geolocation']
        }

    def _save(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Saves processed files to disk."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, df in datasets.items():
            path = self.processed_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.debug(f"Saved {name} to {path}")
