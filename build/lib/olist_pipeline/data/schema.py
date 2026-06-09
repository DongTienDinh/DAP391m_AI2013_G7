from typing import List, Dict
from pydantic import BaseModel, Field

class DatasetSchema(BaseModel):
    """Metadata schema for a dataset."""
    name: str
    required_columns: List[str]
    min_rows: int = 0

# Define expected schemas for core Olist tables
OLIST_SCHEMAS: Dict[str, DatasetSchema] = {
    "orders": DatasetSchema(
        name="orders",
        required_columns=[
            "order_id", "customer_id", "order_status", 
            "order_purchase_timestamp", "order_delivered_customer_date"
        ],
        min_rows=1000
    ),
    "products": DatasetSchema(
        name="products",
        required_columns=[
            "product_id", "product_category_name"
        ],
        min_rows=100
    ),
    "customers": DatasetSchema(
        name="customers",
        required_columns=["customer_id", "customer_unique_id", "customer_state"],
        min_rows=1000
    )
}
