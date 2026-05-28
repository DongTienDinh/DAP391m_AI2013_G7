import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# Cấu hình encoding utf-8 cho stdout trên console Windows để tránh UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm project root vào PYTHONPATH để chạy độc lập không bị lỗi import
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.system_utils import print_section_header


def download_raw_data_if_missing(raw_dir: Path) -> None:
    """
    Kiểm tra nếu dữ liệu thô của Olist chưa tồn tại đầy đủ,
    sẽ tiến hành tải xuống tự động từ Kaggle sử dụng kagglehub và lưu vào raw_dir.
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
        print("-> Dữ liệu thô Olist đã tồn tại đầy đủ trong thư mục raw.")
        return

    print("-> Phát hiện thiếu dữ liệu thô Olist. Tiến hành tải xuống tự động từ Kaggle...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Thực hiện vá lỗi xung đột kagglesdk/kagglehub nếu có
    try:
        import kagglesdk.kaggle_env
        if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(kagglesdk.kaggle_env, 'get_endpoint'):
            kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
    except ImportError:
        pass

    import kagglehub
    downloaded_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    print(f"-> Đang sao chép các tệp dữ liệu từ {downloaded_path} vào {raw_dir}...")
    for file_name in os.listdir(downloaded_path):
        downloaded_file = os.path.join(downloaded_path, file_name)
        dest_file = raw_dir / file_name
        
        if os.path.isfile(downloaded_file):
            shutil.copy(downloaded_file, dest_file)
            print(f"   Đã lưu: {dest_file}")
            
    print("-> Tải xuống dữ liệu thô thành công!")


def load_raw_data(raw_dir: Path) -> dict:
    """
    Đọc tất cả 8 tệp tin CSV thô từ thư mục raw_dir.
    
    Args:
        raw_dir (Path): Thư mục chứa dữ liệu thô từ Kaggle.
        
    Returns:
        dict: Một dictionary chứa các DataFrame tương ứng với tên bảng.
    """
    print(f"-> Đang tải dữ liệu thô từ: {raw_dir}...")
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
        print(f"   Đã tải bảng '{name}': {df.shape[0]:,} dòng × {df.shape[1]} cột")
    return datasets


def clean_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu bảng orders:
    - Chỉ giữ các đơn hàng đã được giao (status == 'delivered').
    - Loại bỏ cột 'order_approved_at'.
    - Loại bỏ các dòng thiếu thông tin ngày giao cho vận chuyển/khách hàng.
    """
    print("-> Đang làm sạch dữ liệu bảng: orders...")
    cleaned_df = (
        orders_df[orders_df['order_status'] == 'delivered']
        .drop(columns=['order_approved_at'])
        .dropna(subset=['order_delivered_carrier_date', 'order_delivered_customer_date'])
        .copy()
    )
    return cleaned_df


def clean_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu bảng products:
    - Loại bỏ dòng thiếu danh mục sản phẩm, trọng lượng, kích thước.
    """
    print("-> Đang làm sạch dữ liệu bảng: products...")
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
    Làm sạch dữ liệu bảng order_reviews:
    - Bỏ cột nội dung bình luận chi tiết để giảm bộ nhớ (title & message).
    """
    print("-> Đang làm sạch dữ liệu bảng: order_reviews...")
    cleaned_df = (
        reviews_df
        .drop(columns=['review_comment_title', 'review_comment_message'])
        .copy()
    )
    return cleaned_df


def save_cleaned_data(datasets: dict, processed_dir: Path) -> None:
    """
    Lưu các bộ dữ liệu đã làm sạch và các bộ dữ liệu thô còn lại sang thư mục processed.
    
    Args:
        datasets (dict): Dictionary chứa các DataFrame đã được tiền xử lý.
        processed_dir (Path): Thư mục đích để lưu trữ.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> Đang lưu trữ dữ liệu sạch vào thư mục: {processed_dir}...")
    
    # 3 bảng được làm sạch cụ thể
    datasets['orders'].to_csv(processed_dir / 'orders.csv', index=False)
    datasets['products'].to_csv(processed_dir / 'products.csv', index=False)
    datasets['order_reviews'].to_csv(processed_dir / 'order_reviews.csv', index=False)
    
    # Các bảng khác được copy trực tiếp từ raw sang processed theo đúng đặc tả notebook
    datasets['customers'].to_csv(processed_dir / 'customers.csv', index=False)
    datasets['order_items'].to_csv(processed_dir / 'order_items.csv', index=False)
    datasets['order_payments'].to_csv(processed_dir / 'order_payments.csv', index=False)
    datasets['sellers'].to_csv(processed_dir / 'sellers.csv', index=False)
    
    print("   Lưu trữ hoàn tất.")


def main():
    print_section_header("BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU")
    
    raw_olist_dir = project_root / "data/raw/olist"
    processed_olist_dir = project_root / "data/processed/olist"
    
    # 0. Tải dữ liệu từ Kaggle nếu chưa tồn tại
    download_raw_data_if_missing(raw_olist_dir)
    
    # 1. Đọc dữ liệu
    raw_datasets = load_raw_data(raw_olist_dir)
    
    # 2. Xử lý làm sạch các bảng
    cleaned_datasets = {}
    cleaned_datasets['orders'] = clean_orders(raw_datasets['orders'])
    cleaned_datasets['products'] = clean_products(raw_datasets['products'])
    cleaned_datasets['order_reviews'] = clean_reviews(raw_datasets['order_reviews'])
    
    # Lưu các bảng khác
    cleaned_datasets['customers'] = raw_datasets['customers']
    cleaned_datasets['order_items'] = raw_datasets['order_items']
    cleaned_datasets['order_payments'] = raw_datasets['order_payments']
    cleaned_datasets['sellers'] = raw_datasets['sellers']
    
    # 3. Thống kê kết quả trước và sau khi làm sạch
    print_section_header("THỐNG KÊ BIẾN ĐỔI DÒNG DỮ LIỆU")
    print(f"  orders   : {len(raw_datasets['orders']):,} -> {len(cleaned_datasets['orders']):,} dòng")
    print(f"  products : {len(raw_datasets['products']):,} -> {len(cleaned_datasets['products']):,} dòng")
    print(f"  reviews  : {raw_datasets['order_reviews'].shape[1]} cột -> {cleaned_datasets['order_reviews'].shape[1]} cột")
    
    print("\n  Kiểm tra giá trị Null còn lại:")
    for name in ['orders', 'products', 'order_reviews']:
        n_nulls = cleaned_datasets[name].isnull().sum().sum()
        status = "⚠️  vẫn còn {} nulls".format(n_nulls) if n_nulls > 0 else "✅  ĐÃ SẠCH"
        print(f"  - Bảng {name:15s}: {status}")
        
    # 4. Ghi dữ liệu sạch ra ổ đĩa
    save_cleaned_data(cleaned_datasets, processed_olist_dir)
    
    print_section_header("QUY TRÌNH HOÀN THÀNH THÀNH CÔNG")


if __name__ == "__main__":
    main()
