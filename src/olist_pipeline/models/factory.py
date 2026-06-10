from typing import Any

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ModelFactory:
    """Factory for creating scikit-learn compatible model pipelines."""

    @staticmethod
    def create_pipeline(
        model_type: str, params: dict[str, Any], random_state: int = 42
    ) -> Pipeline | Any:
        """Creates a pipeline with imputer, scaler and the requested model."""

        if model_type == "ridge":
            model = Ridge(**params, random_state=random_state)
        elif model_type == "elastic_net":
            model = ElasticNet(**params, random_state=random_state)
        elif model_type == "huber":
            model = HuberRegressor(**params)
        elif model_type == "gradient_boosting":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(**params, random_state=random_state)),
                ]
            )
        elif model_type == "random_forest":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(**params, random_state=random_state, n_jobs=-1),
                    ),
                ]
            )
        elif model_type == "xgboost":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(**params, random_state=random_state, n_jobs=-1)),
                ]
            )
        elif model_type == "lightgbm":
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1)),
                ]
            )
        elif model_type == "catboost":
            # CatBoost handles NaNs internally but we keep it consistent for the Zoo
            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", CatBoostRegressor(**params, random_seed=random_state, verbose=0)),
                ]
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
