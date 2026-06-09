from typing import Dict, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

class ModelFactory:
    """Factory for creating scikit-learn compatible model pipelines."""

    @staticmethod
    def create_pipeline(model_type: str, params: Dict[str, Any], random_state: int = 42) -> Union[Pipeline, Any]:
        """Creates a pipeline with scaler and the requested model."""
        
        if model_type == "ridge":
            model = Ridge(**params, random_state=random_state)
        elif model_type == "elastic_net":
            model = ElasticNet(**params, random_state=random_state)
        elif model_type == "huber":
            model = HuberRegressor(**params)
        elif model_type == "random_forest":
            return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        elif model_type == "xgboost":
            return XGBRegressor(**params, random_state=random_state, n_jobs=-1)
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1)
        elif model_type == "catboost":
            return CatBoostRegressor(**params, random_seed=random_state, verbose=0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
