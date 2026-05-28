# -*- coding: utf-8 -*-
"""
Module: expansion_scoring.py
Mục tiêu: Trích xuất, mô-đun hóa và chuyên nghiệp hóa logic tính toán Chỉ số Tiềm năng Mở rộng (Expansion Potential Score - EPS)
          của các bang tại Brazil từ dữ liệu Olist và dữ liệu ngoại cảnh (IBGE population, IBGE GDP).
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

# Thiết lập mã hóa UTF-8 cho stdout/stderr để tránh lỗi hiển thị ký tự tiếng Việt trên Terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Sử dụng chế độ non-interactive 'Agg' cho Matplotlib để tránh lỗi GUI khi chạy dưới dạng script
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import geobr
    HAS_GEOBR = True
except Exception:
    HAS_GEOBR = False

warnings.filterwarnings('ignore')

# Cấu hình logging chuyên nghiệp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Hằng số các bang hợp lệ của Brazil
VALID_STATES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
]

# Cấu hình mặc định
DEFAULT_CONFIG = {
    "rolling_window":            4,
    "use_log_predicted_revenue": True,
    "use_delivered_orders_only": True,
    "growth_clip_lower":         0.01,
    "growth_clip_upper":         0.99,
    "weights": {
        "PD": 0.35,  # Predicted Demand (Dự báo doanh thu)
        "GP": 0.20,  # Growth Potential (Tiềm năng tăng trưởng doanh thu)
        "MS": 0.20,  # Market Size (Quy mô thị trường dựa trên Dân số & GDP)
        "PG": 0.15,  # Penetration Gap (Khoảng cách thâm nhập khách hàng)
        "SI": 0.05,  # Seller Index (Mật độ cạnh tranh nhà bán - phạt điểm)
        "LC": 0.05,  # Logistics Cost (Chi phí & thời gian giao hàng - phạt điểm)
    },
    "logistics_mode": "freight_delivery_average",
}


def get_paths(project_root: Path = None) -> dict:
    """
    Xác định động và trả về các đường dẫn thư mục và tệp tin quan trọng của dự án.
    
    Args:
        project_root: Đường dẫn gốc của dự án. Nếu None, sẽ tự tìm từ thư mục của file này.
        
    Returns:
        dict: Chứa các đối tượng Path quan trọng.
    """
    if project_root is None:
        # Đường dẫn của file hiện tại: src/analysis/expansion_scoring.py -> root là cha của cha của cha (3 levels)
        project_root = Path(__file__).resolve().parents[2]
        
    paths = {
        "root": project_root,
        "processed_olist": project_root / "data" / "processed" / "olist",
        "raw_olist":       project_root / "data" / "raw" / "olist",
        "external":        project_root / "data" / "external",
        "figures_dir":     project_root / "reports" / "figures",
    }
    
    # Tạo thư mục figures nếu chưa có
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    
    # Định nghĩa các đường dẫn tệp cụ thể
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
    Chuẩn hóa Min-Max một Series về khoảng [0, 1].
    Nếu tất cả giá trị bằng nhau, trả về chuỗi toàn số 0.
    """
    s = pd.Series(s).astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mn, mx):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def clip_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Giới hạn giá trị Series theo phân vị (quantile) để loại bỏ nhiễu của các giá trị ngoại lai.
    """
    s = pd.Series(s).astype(float)
    return s.clip(s.quantile(lower), s.quantile(upper))


def parse_week_range(s: str) -> tuple:
    """
    Parse chuỗi khoảng thời gian tuần (ví dụ '2018-08-27/2018-09-02') 
    thành cặp ngày bắt đầu và kết thúc kiểu Timestamp.
    """
    try:
        start, end = str(s).split("/")
        return pd.to_datetime(start), pd.to_datetime(end)
    except Exception as e:
        raise ValueError(f"Không thể parse khoảng tuần '{s}': {e}")


def attach_nearest_past_year(
    panel: pd.DataFrame,
    external: pd.DataFrame,
    value_cols: list,
    state_col: str = "state",
) -> pd.DataFrame:
    """
    Hợp nhất dữ liệu ngoại cảnh (dân số, GDP) vào bảng panel theo năm gần nhất trong quá khứ.
    Giúp xử lý các năm không khớp chính xác giữa Olist (2016-2018) và dữ liệu IBGE.
    """
    external = external.sort_values([state_col, "year"])
    rows = []
    
    # Lấy các cặp bang và năm duy nhất từ panel
    unique_keys = panel[["customer_state", "year"]].drop_duplicates()
    
    for _, row in unique_keys.iterrows():
        st, yr = row["customer_state"], int(row["year"])
        # Tìm các bản ghi của bang này có năm nhỏ hơn hoặc bằng năm hiện tại
        sub = external[(external[state_col] == st) & (external["year"] <= yr)]
        if sub.empty:
            # Nếu không có năm nào trong quá khứ, lấy năm sớm nhất hiện có
            sub = external[external[state_col] == st]
            
        record = {"customer_state": st, "year": yr}
        if sub.empty:
            for c in value_cols:
                record[c] = np.nan
        else:
            # Lấy bản ghi năm lớn nhất gần nhất trong tập lọc
            latest = sub.sort_values("year").iloc[-1]
            for c in value_cols:
                record[c] = latest[c]
        rows.append(record)
        
    rows_df = pd.DataFrame(rows)
    return panel.merge(rows_df, on=["customer_state", "year"], how="left")


def load_table(filename: str, required_cols: list, paths: dict) -> pd.DataFrame:
    """
    Tải một bảng Olist. Ưu tiên tải từ thư mục processed, nếu thiếu cột
    hoặc không tìm thấy sẽ tự động fallback sang thư mục raw.
    """
    processed_path = paths["processed_olist"] / filename
    raw_path       = paths["raw_olist"] / f"olist_{filename.replace('.csv','')}_dataset.csv"

    for path in [processed_path, raw_path]:
        if path.exists():
            df = pd.read_csv(path)
            if all(c in df.columns for c in required_cols):
                logger.info(f"Đã tải thành công '{filename}' từ thư mục '{path.parent.name}'")
                return df

    raise FileNotFoundError(
        f"[ERROR] Không tìm thấy bảng '{filename}' chứa các cột {required_cols} "
        f"ở cả thư mục processed lẫn raw."
    )


def load_data(paths: dict) -> dict:
    """
    Thực hiện tải toàn bộ dữ liệu đầu vào cần thiết cho mô hình tính điểm EPS.
    """
    logger.info("Bắt đầu tải các tệp tin dữ liệu đầu vào...")
    
    # Xác minh các tệp bắt buộc tồn tại
    for key in ["pred_path", "raw_pop", "raw_gdp"]:
        p = paths[key]
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] Thiếu tệp tin bắt buộc: {p}")
            
    # Tải dự báo doanh thu tuần tiếp theo
    pred = pd.read_csv(paths["pred_path"])
    required_pred_cols = ["customer_state", "year_week_current", "predicted_next_week_revenue"]
    missing_cols = set(required_pred_cols) - set(pred.columns)
    if missing_cols:
        raise ValueError(f"[ERROR] Tệp dự báo doanh thu thiếu các cột: {missing_cols}")
        
    # Tải các bảng Olist
    customers = load_table("customers.csv", ["customer_id", "customer_unique_id", "customer_state"], paths)
    orders    = load_table("orders.csv",    ["order_id", "customer_id", "order_status", "order_purchase_timestamp"], paths)
    items     = load_table("order_items.csv", ["order_id", "seller_id", "price", "freight_value"], paths)
    sellers   = load_table("sellers.csv",   ["seller_id", "seller_state"], paths)
    
    # Tải dữ liệu dân số và GDP
    population = pd.read_csv(paths["raw_pop"])
    gdp_df     = pd.read_csv(paths["raw_gdp"])
    
    logger.info(f"Đã tải xong toàn bộ dữ liệu. Tập dự báo doanh thu có {pred.shape[0]} dòng.")
    
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
    Hợp nhất và làm sạch các bảng Olist để tạo bảng Fact giao dịch chi tiết.
    """
    logger.info("Đang xây dựng bảng Fact giao dịch...")
    
    # Xử lý thời gian và tính số ngày giao hàng (delivery time)
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    
    if "order_delivered_customer_date" in orders.columns:
        orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
        orders["delivery_time_days"] = (
            orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400
    else:
        orders["delivery_time_days"] = np.nan

    # Lọc chỉ lấy các đơn hàng đã giao thành công (delivered)
    if "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"].copy()
        
    # Tính doanh thu của từng sản phẩm trong đơn hàng
    items = items.copy()
    if "revenue" not in items.columns:
        items["revenue"] = items["price"] + items["freight_value"]

    # Hợp nhất bảng Fact
    fact = (
        orders
        .merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")
        .merge(items[["order_id", "seller_id", "price", "freight_value", "revenue"]], on="order_id", how="left")
    )
    
    # Lọc các bang hợp lệ và loại bỏ các dòng thiếu trường quan trọng
    fact = fact[fact["customer_state"].isin(VALID_STATES)].copy()
    fact = fact.dropna(subset=["order_purchase_timestamp", "customer_state"])
    
    # Tạo các cột chu kỳ tuần và năm
    fact["week_period"] = fact["order_purchase_timestamp"].dt.to_period("W-SUN")
    fact["week_start"]  = fact["week_period"].apply(lambda p: p.start_time)
    fact["week_end"]    = fact["week_period"].apply(lambda p: p.end_time.normalize())
    fact["year"]        = fact["week_start"].dt.year
    
    logger.info(f"Bảng Fact hoàn thành: {fact.shape[0]:,} dòng giao dịch | "
                f"{fact['customer_state'].nunique()} bang | {fact['week_start'].nunique()} tuần giao dịch.")
    return fact


def aggregate_state_week_features(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Gom nhóm bảng Fact chi tiết để tính toán các đặc trưng tổng hợp cấp bang theo từng tuần.
    Đồng thời reindex để tạo thành lưới thời gian đầy đủ (full panel) không bị thiếu tuần.
    """
    logger.info("Đang gom nhóm và tổng hợp đặc trưng cấp bang theo tuần...")
    
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

    # Tạo dải tuần đầy đủ (Full Grid) từ tuần đầu tiên đến tuần cuối cùng
    all_weeks = pd.date_range(state_week["week_start"].min(), state_week["week_start"].max(), freq="W-MON")
    panel_idx = pd.MultiIndex.from_product(
        [VALID_STATES, all_weeks], names=["customer_state", "week_start"]
    )
    
    # Reindex lưới
    state_week = (
        state_week.set_index(["customer_state", "week_start"])
        .reindex(panel_idx)
        .reset_index()
    )
    
    state_week["week_end"] = state_week["week_start"] + pd.Timedelta(days=6)
    state_week["year"]     = state_week["week_start"].dt.year
    
    # Điền giá trị 0 cho các chỉ số đếm/tổng
    for col in ["revenue", "order_count", "unique_customers", "seller_count"]:
        state_week[col] = state_week[col].fillna(0)
        
    # Nội suy (forward-fill / backward-fill) cho các chỉ số trung bình (giá, cước, thời gian giao)
    for col in ["avg_price", "avg_freight_value", "avg_delivery_time"]:
        state_week[col] = (
            state_week.groupby("customer_state")[col]
            .transform(lambda s: s.ffill().bfill())
        )
        state_week[col] = state_week[col].fillna(state_week[col].median())
        
    logger.info(f"Đã xây dựng xong lưới thời gian bang-tuần (Full Panel): {state_week.shape[0]} dòng.")
    return state_week


def compute_rolling_features(state_week: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Tính toán các đặc trưng rolling (trung bình trượt lịch sử) của bang để tránh rò rỉ dữ liệu.
    """
    logger.info("Đang tính toán các đặc trưng trung bình trượt lịch sử (Rolling Features)...")
    
    state_week = state_week.sort_values(["customer_state", "week_start"]).copy()
    grp = state_week.groupby("customer_state")
    W = config["rolling_window"]

    # 1. Shift doanh thu theo các tuần trước (lags)
    for lag in [1, 2, 4, 8]:
        state_week[f"lag_revenue_{lag}w"] = grp["revenue"].shift(lag)

    # 2. Doanh thu trung bình trượt 4 tuần gần nhất và 4 tuần trước đó để tính Growth Potential
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

    # 3. Số khách hàng duy nhất tích lũy trượt 4 tuần
    state_week["rolling_customers_4w"] = grp["unique_customers"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).sum()
    )
    
    # 4. Số lượng người bán trung bình trượt 4 tuần
    state_week["rolling_sellers_4w"] = grp["seller_count"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )

    # 5. Đặc trưng logistics (cước phí và thời gian giao hàng trung bình trượt 4 tuần)
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
    Tính toán Chỉ số Tiềm năng Mở rộng (EPS) tổng hợp và xếp hạng các bang.
    """
    logger.info("Đang ghép nối dữ liệu dự báo và tính toán chỉ số EPS...")
    
    # Chuẩn hóa dữ liệu dân số
    population = population.copy()
    population["state"] = population["state"].astype(str).str.strip().str.upper()
    population = population[population["state"].isin(VALID_STATES)].copy()
    population = population.dropna(subset=["year", "state", "population"])
    population["year"] = population["year"].astype(int)
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    population = population[population["population"] > 0]
    
    # Chuẩn hóa dữ liệu GDP
    gdp_df = gdp_df.copy()
    gdp_df["state"] = gdp_df["state"].astype(str).str.strip().str.upper()
    gdp_df = gdp_df[gdp_df["state"].isin(VALID_STATES)].copy()
    gdp_df = gdp_df.dropna(subset=["year", "state", "gdp"])
    gdp_df["year"] = gdp_df["year"].astype(int)
    gdp_df["gdp"] = pd.to_numeric(gdp_df["gdp"], errors="coerce")
    gdp_df = gdp_df[gdp_df["gdp"] > 0]

    # Ghép nối dân số và GDP vào state_week
    state_week = attach_nearest_past_year(state_week, population, value_cols=["population"])
    state_week = attach_nearest_past_year(state_week, gdp_df, value_cols=["gdp"])
    state_week["gdp_per_capita"] = state_week["gdp"] / state_week["population"]

    # Phân tích ngày bắt đầu/kết thúc từ tuần dự báo
    pred = pred.copy()
    pred[["current_week_start", "current_week_end"]] = pred["year_week_current"].apply(
        lambda x: pd.Series(parse_week_range(x))
    )

    # Ghép nối tập dự báo với tập đặc trưng tuần
    eps = pred.merge(
        state_week,
        left_on  = ["customer_state", "current_week_start"],
        right_on = ["customer_state", "week_start"],
        how="left"
    )

    # Tự động khắc phục nếu lệch chu kỳ tuần (ví dụ Mon vs Sun)
    if eps["revenue"].isna().all():
        logger.info("[WARN] Phát hiện lệch tuần giữa dự báo và panel, đang căn chỉnh lại chu kỳ...")
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

    # Join bổ sung dân số/GDP nếu chưa có
    if "population" not in eps.columns:
        eps = attach_nearest_past_year(eps, population, ["population"])
    if "gdp" not in eps.columns:
        eps = attach_nearest_past_year(eps, gdp_df, ["gdp"])
        
    eps["gdp_per_capita"] = eps["gdp"] / eps["population"]

    # Kiểm tra tính toàn vẹn dữ liệu ngoại cảnh
    assert eps.shape[0] == pred.shape[0], "[ERROR] Số lượng dòng không khớp sau khi merge."
    assert eps["population"].notna().all(), "[ERROR] Có bang bị thiếu dữ liệu Dân số."
    assert eps["gdp"].notna().all(), "[ERROR] Có bang bị thiếu dữ liệu GDP."

    # ──── TÍNH TOÁN CÁC THÀNH PHẦN EPS ────
    # 1. PD (Predicted Demand): Nhu cầu dự kiến của tuần tiếp theo (lấy Log1p)
    eps["PD_raw"] = np.log1p(eps["predicted_next_week_revenue"])

    # 2. GP (Growth Potential): Tiềm năng tăng trưởng doanh thu trượt
    eps["GP_raw"] = clip_series(
        eps["growth_potential_raw"],
        lower=config["growth_clip_lower"],
        upper=config["growth_clip_upper"],
    )

    # 3. MS (Market Size): Quy mô thị trường (Dân số kết hợp GDP bình quân đầu người)
    national_avg_gdppc = eps["gdp"].sum() / eps["population"].sum()
    eps["gdp_per_capita_index"] = eps["gdp_per_capita"] / national_avg_gdppc
    eps["MS_raw"] = np.log1p(eps["population"]) * eps["gdp_per_capita_index"]

    # 4. PG (Penetration Gap): Chênh lệch độ thâm nhập (Quy mô thị trường trừ đi tỉ lệ khách hàng hiện tại)
    eps["MS_norm_temp"]          = minmax_series(eps["MS_raw"])
    eps["penetration_norm_temp"] = minmax_series(
        eps["rolling_customers_4w"] / eps["population"]
    )
    eps["PG_raw"] = eps["MS_norm_temp"] - eps["penetration_norm_temp"]

    # 5. SI (Seller Index): Mật độ cạnh tranh người bán (số người bán trên 100k dân)
    eps["SI_raw"] = eps["rolling_sellers_4w"] / eps["population"] * 100_000

    # 6. LC (Logistics Cost): Chỉ số logistics (phối hợp cước phí và thời gian giao hàng)
    if config["logistics_mode"] == "freight_delivery_average":
        eps["LC_raw"] = (
            0.5 * minmax_series(eps["avg_freight_4w"])
            + 0.5 * minmax_series(eps["avg_delivery_time_4w"])
        )
    else:
        eps["LC_raw"] = minmax_series(eps["avg_freight_4w"])

    # Chuẩn hóa Min-Max tất cả các thành phần về dải [0, 1]
    for raw_col in ["PD_raw", "GP_raw", "MS_raw", "PG_raw", "SI_raw", "LC_raw"]:
        norm_col = raw_col.replace("_raw", "_norm")
        eps[norm_col] = minmax_series(eps[raw_col])

    # Tính điểm EPS tổng hợp có trọng số
    W = config["weights"]
    eps["EPS"] = (
         W["PD"] * eps["PD_norm"]
      +  W["GP"] * eps["GP_norm"]
      +  W["MS"] * eps["MS_norm"]
      +  W["PG"] * eps["PG_norm"]
      -  W["SI"] * eps["SI_norm"]
      -  W["LC"] * eps["LC_norm"]
    )

    # Chuẩn hóa điểm EPS về thang điểm [0, 100]
    eps["EPS_0_100"] = 100 * (
        (eps["EPS"] - eps["EPS"].min())
        / (eps["EPS"].max() - eps["EPS"].min() + 1e-9)
    )
    
    # Xếp hạng bang dựa trên điểm EPS
    eps["rank_eps"] = eps["EPS_0_100"].rank(ascending=False, method="dense").astype(int)
    
    # Tạo cột giải thích ý nghĩa (interpretation)
    eps["interpretation"] = eps.apply(interpret_eps, axis=1)
    
    logger.info("Đã hoàn thành tính toán điểm số và xếp hạng EPS.")
    return eps


def interpret_eps(row) -> str:
    """
    Phân tích các thành phần để đưa ra giải thích lý do bang đạt điểm EPS cao/thấp.
    """
    reasons = []
    if row["PD_norm"] >= 0.75: reasons.append("nhu cầu dự báo cao")
    if row["GP_norm"] >= 0.75: reasons.append("tăng trưởng mạnh")
    if row["MS_norm"] >= 0.75: reasons.append("quy mô thị trường lớn")
    if row["PG_norm"] >= 0.75: reasons.append("còn nhiều dư địa thâm nhập")
    if row["SI_norm"] >= 0.75: reasons.append("mật độ cạnh tranh bán hàng cao")
    if row["LC_norm"] >= 0.75: reasons.append("chi phí logistics cao")
    
    return ", ".join(reasons) if reasons else "hồ sơ phát triển cân bằng"


def save_outputs(eps_df: pd.DataFrame, paths: dict) -> None:
    """
    Lưu trữ các tệp tin kết quả đầu ra CSV.
    """
    # 1. Lưu tập dữ liệu đặc trưng EPS đầy đủ
    eps_df.to_csv(paths["output_features"], index=False)
    logger.info(f"Đã lưu tệp đặc trưng EPS: {paths['output_features']}")

    # 2. Lọc các cột cần thiết cho bảng xếp hạng và lưu
    ranking_cols = [
        "customer_state", "year_week_current", "predicted_next_week_revenue",
        "PD_norm", "GP_norm", "MS_norm", "PG_norm", "SI_norm", "LC_norm",
        "EPS", "EPS_0_100", "rank_eps", "interpretation"
    ]
    eps_ranking = eps_df[ranking_cols].sort_values("rank_eps").reset_index(drop=True)
    eps_ranking.to_csv(paths["output_ranking"], index=False)
    logger.info(f"Đã lưu bảng xếp hạng tiềm năng mở rộng: {paths['output_ranking']}")


def plot_visualizations(eps_ranking: pd.DataFrame, paths: dict) -> None:
    """
    Vẽ và lưu trữ các biểu đồ trực quan hóa kết quả:
    1. Biểu đồ thanh ngang Top 15 bang có tiềm năng mở rộng lớn nhất.
    2. Bản đồ nhiệt phân bố cơ hội (Choropleth Heatmap) trên đất nước Brazil.
    """
    logger.info("Đang tạo các biểu đồ báo cáo trực quan...")
    
    # 1. Vẽ biểu đồ thanh ngang
    top_states = eps_ranking.sort_values("EPS_0_100", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top_states["customer_state"][::-1],
        top_states["EPS_0_100"][::-1],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_states))),
        edgecolor="white",
    )
    ax.set_xlabel("Điểm Tiềm Năng EPS (0–100)", fontsize=10)
    ax.set_ylabel("Bang", fontsize=10)
    ax.set_title("Top 15 bang của Brazil có tiềm năng mở rộng thị trường lớn nhất (EPS)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    
    # Ghi nhãn điểm số bên cạnh mỗi cột thanh
    for bar, v in zip(bars, top_states["EPS_0_100"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=8, fontweight="bold")
                
    plt.tight_layout()
    bar_chart_path = paths["figures_dir"] / "eps_state_ranking_bar.png"
    plt.savefig(bar_chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Đã lưu biểu đồ thanh ngang tại: {bar_chart_path}")

    # 2. Vẽ bản đồ nhiệt Choropleth
    states_geo = None
    
    # Ưu tiên load từ tệp GeoJSON cục bộ
    if paths["br_states_json"].exists():
        try:
            states_geo = gpd.read_file(paths["br_states_json"])
            logger.info("Đã tải thành công bản đồ Brazil từ tệp GeoJSON cục bộ.")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc file GeoJSON cục bộ: {e}. Thử fallback sang geobr...")
            
    # Fallback sang thư viện geobr
    if states_geo is None and HAS_GEOBR:
        try:
            states_geo = geobr.read_state(year=2018)
            logger.info("Đã tải thành công bản đồ Brazil từ thư viện geobr trực tuyến.")
        except Exception as e:
            logger.warning(f"Lỗi khi tải bản đồ bằng geobr: {e}. Thử fallback sang file shapefile địa phương...")

    # Fallback cuối cùng sang shapefiles cục bộ
    if states_geo is None:
        shp_files = list(paths["raw_geobr"].rglob("*.shp"))
        if not shp_files:
            logger.error("Không tìm thấy bất kỳ bản đồ Brazil nào để vẽ heatmap.")
            return
        
        # Ưu tiên tệp chứa ranh giới bang (states)
        states_shp = [s for s in shp_files if "states" in s.name]
        selected_shp = states_shp[0] if states_shp else shp_files[0]
        try:
            states_geo = gpd.read_file(selected_shp)
            logger.info(f"Đã tải thành công bản đồ từ shapefile cục bộ: {selected_shp.name}")
        except Exception as e:
            logger.error(f"Không thể đọc file shapefile: {e}")
            return

    # Chuẩn hóa tên cột bang của bản đồ về 'customer_state' để merge
    state_col_candidates = ["abbrev_state", "sigla_uf", "SIGLA_UF", "UF", "uf"]
    matched = [c for c in state_col_candidates if c in states_geo.columns]
    if not matched:
        logger.error(f"Không tìm thấy cột viết tắt tên bang. Cột hiện có: {states_geo.columns.tolist()}")
        return
        
    states_geo = states_geo.rename(columns={matched[0]: "customer_state"})
    
    # Merge bản đồ với bảng xếp hạng EPS
    gdf = states_geo.merge(eps_ranking[["customer_state", "EPS_0_100", "rank_eps"]],
                           on="customer_state", how="left")

    # Vẽ và xuất bản đồ
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column="EPS_0_100", ax=ax, legend=True, cmap="YlOrRd",
        edgecolor="black", linewidth=0.4,
        missing_kwds={"color": "lightgrey", "label": "Không có dữ liệu"},
        legend_kwds={'label': "Expansion Potential Score (EPS)", 'orientation': "horizontal"}
    )
    ax.set_title("Bản đồ Tiềm năng Mở rộng (Expansion Potential Score) — Brazil States", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    
    heatmap_path = paths["figures_dir"] / "eps_state_heatmap.png"
    plt.savefig(heatmap_path, dpi=250, bbox_inches="tight")
    plt.close()
    logger.info(f"Đã lưu bản đồ nhiệt tại: {heatmap_path}")


def run_scoring_pipeline(config_override: dict = None, project_root: Path = None) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline tính điểm Expansion Potential Score từ đầu đến cuối.
    
    Args:
        config_override: Từ điển ghi đè cấu hình mặc định (ví dụ trọng số hoặc rolling window).
        project_root: Thư mục gốc của dự án.
        
    Returns:
        pd.DataFrame: Bảng chứa dữ liệu EPS hoàn chỉnh của các bang.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU PIPELINE TÍNH ĐIỂM TIỀM NĂNG MỞ RỘNG (EPS)")
    logger.info("=" * 60)
    
    # Áp dụng cấu hình và cập nhật nếu có ghi đè
    config = dict(DEFAULT_CONFIG)
    if config_override:
        if "weights" in config_override and isinstance(config_override["weights"], dict):
            config["weights"].update(config_override["weights"])
        # Cập nhật các tham số cấu hình khác
        for k, v in config_override.items():
            if k != "weights":
                config[k] = v
                
    logger.info(f"Cấu hình áp dụng: {config}")

    # Xác định đường dẫn các tệp
    paths = get_paths(project_root)
    
    # 1. Tải dữ liệu đầu vào
    data = load_data(paths)
    
    # 2. Xây dựng bảng Fact giao dịch
    fact = build_fact_table(data["orders"], data["customers"], data["items"])
    
    # 3. Gom nhóm đặc trưng cấp bang theo tuần
    state_week = aggregate_state_week_features(fact)
    
    # 4. Tính toán đặc trưng trượt lịch sử (Rolling features)
    state_week = compute_rolling_features(state_week, config)
    
    # 5. Tính điểm EPS tổng hợp
    eps_df = calculate_eps(data["pred"], state_week, data["population"], data["gdp"], config)
    
    # 6. Lưu kết quả ra tệp CSV
    save_outputs(eps_df, paths)
    
    # 7. Trực quan hóa kết quả
    ranking_cols = [
        "customer_state", "year_week_current", "predicted_next_week_revenue",
        "PD_norm", "GP_norm", "MS_norm", "PG_norm", "SI_norm", "LC_norm",
        "EPS", "EPS_0_100", "rank_eps", "interpretation"
    ]
    eps_ranking = eps_df[ranking_cols].sort_values("rank_eps").reset_index(drop=True)
    plot_visualizations(eps_ranking, paths)
    
    logger.info("=" * 60)
    logger.info("PIPELINE EPS HOÀN THÀNH THÀNH CÔNG!")
    logger.info("=" * 60)
    
    return eps_df


if __name__ == "__main__":
    # Phân tích tham số dòng lệnh CLI
    parser = argparse.ArgumentParser(description="Chạy pipeline tính điểm tiềm năng mở rộng các bang (EPS)")
    parser.add_argument("--project-root", type=str, default=None, help="Đường dẫn gốc của dự án")
    parser.add_argument("--rolling-window", type=int, default=4, help="Thời gian trượt (số tuần)")
    parser.add_argument("--w-pd", type=float, default=0.35, help="Trọng số cho PD (Predicted Demand)")
    parser.add_argument("--w-gp", type=float, default=0.20, help="Trọng số cho GP (Growth Potential)")
    parser.add_argument("--w-ms", type=float, default=0.20, help="Trọng số cho MS (Market Size)")
    parser.add_argument("--w-pg", type=float, default=0.15, help="Trọng số cho PG (Penetration Gap)")
    parser.add_argument("--w-si", type=float, default=0.05, help="Trọng số cho SI (Seller Index)")
    parser.add_argument("--w-lc", type=float, default=0.05, help="Trọng số cho LC (Logistics Cost)")
    
    args = parser.parse_args()
    
    # Xây dựng cấu hình ghi đè từ CLI
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
        logger.exception(f"[FATAL ERROR] Lỗi không thể phục hồi khi chạy pipeline: {e}")
        sys.exit(1)
