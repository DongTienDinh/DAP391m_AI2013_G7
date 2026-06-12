from typing import Any

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
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

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(**params, random_state=random_state)
        elif model_type == "elastic_net":
            model = ElasticNet(**params, random_state=random_state)
        elif model_type == "huber":
            model = HuberRegressor(**params)
        elif model_type == "gradient_boosting":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(**params, random_state=random_state)),
            ])
        elif model_type == "random_forest":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)),
            ])
        elif model_type == "xgboost":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(**params, random_state=random_state, n_jobs=-1)),
            ])
        elif model_type == "lightgbm":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1)),
            ])
        elif model_type == "catboost":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostRegressor(**params, random_seed=random_state, verbose=0)),
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    @staticmethod
    def create_all(
        model_configs: dict[str, dict[str, Any]], random_state: int = 42
    ) -> dict[str, Any]:
        """Creates all defined models as a name -> pipeline dict."""
        models = {}
        for model_type, params in model_configs.items():
            name = {
                "linear": "Linear Regression (Baseline)",
                "ridge": "Ridge Regression",
                "elastic_net": "ElasticNet",
                "huber": "Huber Regressor",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM",
                "catboost": "CatBoost",
            }.get(model_type, model_type)
            models[name] = ModelFactory.create_pipeline(model_type, params, random_state)
        return models
