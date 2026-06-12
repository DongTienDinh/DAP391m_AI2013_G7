import os
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from src.core.exceptions import DataIngestionError
from src.core.logging_setup import setup_logger

logger = setup_logger("data_loader")


class RawDataLoader:
    """Handles downloading and loading raw Olist data."""

    REQUIRED_FILES = [
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
    ]

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def download_if_missing(self) -> None:
        """Downloads data from Kaggle if any required file is missing."""
        missing = [f for f in self.REQUIRED_FILES if not (self.raw_dir / f).exists()]

        if not missing:
            logger.info("All raw Olist files present.")
            return

        logger.info(f"Missing {len(missing)} files. Downloading from Kaggle...")
        try:
            downloaded_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
            self.raw_dir.mkdir(parents=True, exist_ok=True)

            for file_name in os.listdir(downloaded_path):
                src = os.path.join(downloaded_path, file_name)
                dst = self.raw_dir / file_name
                if os.path.isfile(src):
                    shutil.copy(src, dst)
            logger.info("Download and extraction completed successfully.")
        except Exception as e:
            raise DataIngestionError(f"Kaggle download failed: {e}")

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Loads all Olist CSVs into a dictionary of DataFrames."""
        datasets = {}
        # Mapping from file name parts to dictionary keys
        mapping = {
            "customers": "olist_customers_dataset.csv",
            "geolocation": "olist_geolocation_dataset.csv",
            "orders": "olist_orders_dataset.csv",
            "order_items": "olist_order_items_dataset.csv",
            "order_payments": "olist_order_payments_dataset.csv",
            "order_reviews": "olist_order_reviews_dataset.csv",
            "products": "olist_products_dataset.csv",
            "sellers": "olist_sellers_dataset.csv",
        }

        for key, file_name in mapping.items():
            path = self.raw_dir / file_name
            if not path.exists():
                raise DataIngestionError(f"Missing required file: {path}")
            datasets[key] = pd.read_csv(path)
            logger.debug(f"Loaded {key}: {datasets[key].shape}")

        return datasets
