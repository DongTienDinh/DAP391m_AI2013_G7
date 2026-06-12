
import pandas as pd
import pytest

from src.core.config import DataPaths


@pytest.fixture
def mock_path_config(tmp_path):
    """Fixture for temporary directory structure."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    return DataPaths(raw_olist=raw, processed_olist=processed)

@pytest.fixture
def sample_feature_df():
    """Fixture for a small dummy feature matrix."""
    return pd.DataFrame({
        "customer_state": ["SP", "SP", "RJ", "RJ"],
        "year_week": pd.to_datetime(["2018-01-01", "2018-01-08", "2018-01-01", "2018-01-08"]).to_period('W'),
        "revenue": [100.0, 150.0, 50.0, 60.0],
        "month": [1, 1, 1, 1]
    })
