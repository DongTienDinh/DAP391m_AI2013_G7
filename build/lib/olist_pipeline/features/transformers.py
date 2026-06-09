import numpy as np
import pandas as pd
from typing import List

class FeatureTransformers:
    """Static utility class for common feature transformations."""

    @staticmethod
    def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
        """Applies cyclical sin/cos encoding to temporal attributes."""
        df = df.copy()
        week_of_year = df['year_week'].dt.week.astype(float)
        df['week_of_year'] = week_of_year.astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        df['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        return df

    @staticmethod
    def add_lags_and_rolling(df: pd.DataFrame, target_col: str = 'revenue') -> pd.DataFrame:
        """Calculates growth rates, lagged revenue, and rolling window statistics."""
        df = df.sort_values(['customer_state', 'year_week']).reset_index(drop=True)
        grp = df.groupby('customer_state')
        
        # Growth
        df[f'{target_col}_growth_1w'] = grp[target_col].pct_change(1).fillna(0)
        df[f'{target_col}_growth_4w'] = grp[target_col].pct_change(4).fillna(0)
        
        # Lags
        for lag in [1, 2, 4, 8]:
            df[f'{target_col}_lag_{lag}'] = grp[target_col].shift(lag)
            
        # Rolling Windows
        for window in [4, 8, 12]:
            df[f'{target_col}_rolling_{window}'] = grp[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).mean()
            )
            
        # Target (shifted back)
        df['target_next_revenue'] = grp[target_col].shift(-1)
        
        return df
