import sys
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict

from src.olist_pipeline.utils.system_utils import print_section_header
from src.olist_pipeline.utils.logger import setup_logger

logger = setup_logger("data_cleaning")

def download_raw_data_if_missing(raw_dir: Path) -> None:
    """
    Checks for Olist raw data and downloads from Kaggle if missing.
    """
    required_files = [
        'olist_customers_dataset.csv',
        'olist_geolocation_dataset.csv',
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_order_payments_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv'
    ]
    
    missing_files = []
    if not raw_dir.exists():
        missing_files = required_files
    else:
        for f in required_files:
            if not (raw_dir / f).exists():
                missing_files.append(f)
                
    if not missing_files:
        logger.info("Olist raw data already fully exists in raw directory.")
        return

    logger.info("Missing Olist raw data detected. Downloading from Kaggle...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import kagglesdk.kaggle_env
        if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(kagglesdk.kaggle_env, 'get_endpoint'):
            kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
    except ImportError:
        pass

    import kagglehub
    downloaded_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    logger.info(f"Copying data files from {downloaded_path} into {raw_dir}...")
    for file_name in os.listdir(downloaded_path):
        downloaded_file = os.path.join(downloaded_path, file_name)
        dest_file = raw_dir / file_name
        
        if os.path.isfile(downloaded_file):
            shutil.copy(downloaded_file, dest_file)
            logger.info(f"   Saved: {dest_file}")
            
    logger.info("Downloaded raw data successfully!")


def load_raw_data(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Loads all 8 raw Olist CSV files into a dictionary of DataFrames.
    """
    logger.info(f"Loading raw data from: {raw_dir}...")
    datasets = {
        'customers': pd.read_csv(raw_dir / 'olist_customers_dataset.csv'),
        'geolocation': pd.read_csv(raw_dir / 'olist_geolocation_dataset.csv'),
        'orders': pd.read_csv(raw_dir / 'olist_orders_dataset.csv'),
        'order_items': pd.read_csv(raw_dir / 'olist_order_items_dataset.csv'),
        'order_payments': pd.read_csv(raw_dir / 'olist_order_payments_dataset.csv'),
        'order_reviews': pd.read_csv(raw_dir / 'olist_order_reviews_dataset.csv'),
        'products': pd.read_csv(raw_dir / 'olist_products_dataset.csv'),
        'sellers': pd.read_csv(raw_dir / 'olist_sellers_dataset.csv')
    }
    for name, df in datasets.items():
        logger.info(f"   Loaded table '{name}': {df.shape[0]:,} rows")
    return datasets


def clean_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for delivered orders and removes invalid delivery dates.
    """
    logger.info("Cleaning orders table...")
    return (
        orders_df[orders_df['order_status'] == 'delivered']
        .drop(columns=['order_approved_at'])
        .dropna(subset=['order_delivered_carrier_date', 'order_delivered_customer_date'])
        .copy()
    )


def clean_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes products with missing categories or physical dimensions.
    """
    logger.info("Cleaning products table...")
    return (
        products_df
        .dropna(subset=[
            'product_category_name',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm'
        ])
        .copy()
    )


def clean_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops review text columns to save memory.
    """
    logger.info("Cleaning reviews table...")
    return (
        reviews_df
        .drop(columns=['review_comment_title', 'review_comment_message'])
        .copy()
    )


def save_cleaned_data(datasets: Dict[str, pd.DataFrame], processed_dir: Path) -> None:
    """
    Saves DataFrames to CSV in the processed directory.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving cleaned data to: {processed_dir}...")
    
    for name, df in datasets.items():
        file_name = f"{name}.csv"
        df.to_csv(processed_dir / file_name, index=False)
    
    logger.info("   Storage completed.")

def run_cleaning_pipeline(raw_dir: Path, processed_dir: Path) -> None:
    """
    Executes the full cleaning pipeline.
    """
    print_section_header("STARTING DATA CLEANING PIPELINE")
    
    download_raw_data_if_missing(raw_dir)
    raw_datasets = load_raw_data(raw_dir)
    
    cleaned_datasets = {
        'orders': clean_orders(raw_datasets['orders']),
        'products': clean_products(raw_datasets['products']),
        'order_reviews': clean_reviews(raw_datasets['order_reviews']),
        'customers': raw_datasets['customers'],
        'order_items': raw_datasets['order_items'],
        'order_payments': raw_datasets['order_payments'],
        'sellers': raw_datasets['sellers']
    }
    
    print_section_header("DATA TRANSFORMATION SUMMARY")
    logger.info(f"  Orders   : {len(raw_datasets['orders']):,} -> {len(cleaned_datasets['orders']):,} rows")
    logger.info(f"  Products : {len(raw_datasets['products']):,} -> {len(cleaned_datasets['products']):,} rows")
    
    save_cleaned_data(cleaned_datasets, processed_dir)
    print_section_header("PIPELINE COMPLETED")
