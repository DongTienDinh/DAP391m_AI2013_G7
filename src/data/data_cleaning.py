import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# Configure utf-8 encoding for stdout on Windows console to avoid UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to PYTHONPATH for standalone execution to avoid import errors
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.system_utils import print_section_header


def download_raw_data_if_missing(raw_dir: Path) -> None:
    """
    Check if Olist raw data does not fully exist,
    automatically download it from Kaggle using kagglehub and save to raw_dir.
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
        print("-> Olist raw data already fully exists in raw directory.")
        return

    print("-> Missing Olist raw data detected. Downloading automatically from Kaggle...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply hotfix for kagglesdk/kagglehub conflicts if any
    try:
        import kagglesdk.kaggle_env
        if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(kagglesdk.kaggle_env, 'get_endpoint'):
            kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
    except ImportError:
        pass

    import kagglehub
    downloaded_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    print(f"-> Copying data files from {downloaded_path} into {raw_dir}...")
    for file_name in os.listdir(downloaded_path):
        downloaded_file = os.path.join(downloaded_path, file_name)
        dest_file = raw_dir / file_name
        
        if os.path.isfile(downloaded_file):
            shutil.copy(downloaded_file, dest_file)
            print(f"   Saved: {dest_file}")
            
    print("-> Downloaded raw data successfully!")


def load_raw_data(raw_dir: Path) -> dict:
    """
    Read all 8 raw CSV files from raw_dir.
    
    Args:
        raw_dir (Path): Directory containing raw data from Kaggle.
        
    Returns:
        dict: A dictionary containing DataFrames corresponding to table names.
    """
    print(f"-> Loading raw data from: {raw_dir}...")
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
        print(f"   Loaded table '{name}': {df.shape[0]:,} rows × {df.shape[1]} columns")
    return datasets


def clean_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean orders table data:
    - Only keep delivered orders (status == 'delivered').
    - Drop 'order_approved_at' column.
    - Drop rows missing carrier/customer delivery date.
    """
    print("-> Cleaning data for table: orders...")
    cleaned_df = (
        orders_df[orders_df['order_status'] == 'delivered']
        .drop(columns=['order_approved_at'])
        .dropna(subset=['order_delivered_carrier_date', 'order_delivered_customer_date'])
        .copy()
    )
    return cleaned_df


def clean_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean products table data:
    - Drop rows missing product category, weight, or dimensions.
    """
    print("-> Cleaning data for table: products...")
    cleaned_df = (
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
    return cleaned_df


def clean_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean order_reviews table data:
    - Drop detailed review comment columns to save memory (title & message).
    """
    print("-> Cleaning data for table: order_reviews...")
    cleaned_df = (
        reviews_df
        .drop(columns=['review_comment_title', 'review_comment_message'])
        .copy()
    )
    return cleaned_df


def save_cleaned_data(datasets: dict, processed_dir: Path) -> None:
    """
    Save cleaned datasets and remaining raw datasets to the processed directory.
    
    Args:
        datasets (dict): Dictionary containing preprocessed DataFrames.
        processed_dir (Path): Target directory for storage.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> Saving clean data to directory: {processed_dir}...")
    
    # 3 specifically cleaned tables
    datasets['orders'].to_csv(processed_dir / 'orders.csv', index=False)
    datasets['products'].to_csv(processed_dir / 'products.csv', index=False)
    datasets['order_reviews'].to_csv(processed_dir / 'order_reviews.csv', index=False)
    
    # Other tables copied directly from raw to processed as per notebook specs
    datasets['customers'].to_csv(processed_dir / 'customers.csv', index=False)
    datasets['order_items'].to_csv(processed_dir / 'order_items.csv', index=False)
    datasets['order_payments'].to_csv(processed_dir / 'order_payments.csv', index=False)
    datasets['sellers'].to_csv(processed_dir / 'sellers.csv', index=False)
    
    print("   Storage completed.")


def main():
    print_section_header("STARTING PREPROCESSING & DATA CLEANING PIPELINE")
    
    raw_olist_dir = project_root / "data/raw/olist"
    processed_olist_dir = project_root / "data/processed/olist"
    
    # 0. Download data from Kaggle if it doesn't exist
    download_raw_data_if_missing(raw_olist_dir)
    
    # 1. Load data
    raw_datasets = load_raw_data(raw_olist_dir)
    
    # 2. Clean tables
    cleaned_datasets = {}
    cleaned_datasets['orders'] = clean_orders(raw_datasets['orders'])
    cleaned_datasets['products'] = clean_products(raw_datasets['products'])
    cleaned_datasets['order_reviews'] = clean_reviews(raw_datasets['order_reviews'])
    
    # Save other tables
    cleaned_datasets['customers'] = raw_datasets['customers']
    cleaned_datasets['order_items'] = raw_datasets['order_items']
    cleaned_datasets['order_payments'] = raw_datasets['order_payments']
    cleaned_datasets['sellers'] = raw_datasets['sellers']
    
    # 3. Statistics of results before and after cleaning
    print_section_header("DATA ROW TRANSFORMATION STATISTICS")
    print(f"  orders   : {len(raw_datasets['orders']):,} -> {len(cleaned_datasets['orders']):,} rows")
    print(f"  products : {len(raw_datasets['products']):,} -> {len(cleaned_datasets['products']):,} rows")
    print(f"  reviews  : {raw_datasets['order_reviews'].shape[1]} columns -> {cleaned_datasets['order_reviews'].shape[1]} columns")
    
    print("\n  Check remaining Null values:")
    for name in ['orders', 'products', 'order_reviews']:
        n_nulls = cleaned_datasets[name].isnull().sum().sum()
        status = "⚠️  still has {} nulls".format(n_nulls) if n_nulls > 0 else "✅  CLEANED"
        print(f"  - Table {name:15s}: {status}")
        
    # 4. Write clean data to disk
    save_cleaned_data(cleaned_datasets, processed_olist_dir)
    
    print_section_header("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
