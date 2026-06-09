import pandas as pd

from src.olist_pipeline.features.transformers import FeatureTransformers


def test_add_seasonality(sample_feature_df):
    """Verifies sin/cos components are added."""
    df = FeatureTransformers.add_seasonality(sample_feature_df)

    assert "month_sin" in df.columns
    assert "month_cos" in df.columns
    assert "week_sin" in df.columns
    assert len(df) == len(sample_feature_df)

def test_add_lags(sample_feature_df):
    """Verifies lag columns are generated."""
    df = FeatureTransformers.add_lags_and_rolling(sample_feature_df)

    assert "revenue_lag_1" in df.columns
    assert "target_next_revenue" in df.columns
    # First week of SP should have NaN for lag_1
    assert pd.isna(df[df['customer_state'] == 'SP'].iloc[0]['revenue_lag_1'])
