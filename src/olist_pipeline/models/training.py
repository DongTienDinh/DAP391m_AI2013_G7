import os
import sys
import io
import time
import warnings
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model     import LinearRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import TimeSeriesSplit
from sklearn.metrics          import mean_squared_error, mean_absolute_error
from xgboost                  import XGBRegressor
import lightgbm as lgb
from catboost                 import CatBoostRegressor

from src.olist_pipeline.utils.logger import setup_logger
from src.olist_pipeline.utils.system_utils import print_section_header

logger = setup_logger("model_training")

warnings.filterwarnings('ignore')

# Academic plotting style configuration
ACADEMIC_RC = {
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
}

MODEL_SHORT_NAMES = {
    "Linear Regression (Baseline)": "Linear",
    "Ridge Regression":             "Ridge",
    "ElasticNet":                   "ElasticNet",
    "Huber Regressor":              "Huber",
    "Random Forest":                "RF",
    "Gradient Boosting":            "GBR",
    "XGBoost":                      "XGBoost",
    "LightGBM":                     "LGBM",
    "CatBoost":                     "CatBoost",
}


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """
    Loads weekly feature data and prepares it for modeling.
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Encode state codes for tree models
    df['state_code'] = df['customer_state'].astype('category').cat.codes
    
    # Temporal sort for walk-forward CV
    df = df.sort_values('year_week').reset_index(drop=True)
    return df


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rolling volatility and momentum features.
    """
    grp = df.groupby('customer_state')
    df['revenue_std_4']   = grp['revenue'].transform(lambda x: x.shift(1).rolling(4, min_periods=2).std()).fillna(0)
    df['revenue_momentum'] = (df['revenue_lag_1'] - df['revenue_lag_4']).fillna(0)
    return df


def select_features(df: pd.DataFrame, exclude_cols: List[str], log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Selects model features and applies log transformation to target.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target_next_revenue'].values
    
    if log_target:
        y_model = np.log1p(y)
        logger.info(f"Target log-transformed. New skewness: {pd.Series(y_model).skew():.3f}")
    else:
        y_model = y
        
    return X, y, y_model, feature_cols


def get_model_definitions(model_configs: Dict[str, Any], random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of initialized model pipelines.
    """
    return {
        "Linear Regression (Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=model_configs['ridge']['alpha'], random_state=random_state))
        ]),
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=model_configs['elastic_net']['alpha'], l1_ratio=model_configs['elastic_net']['l1_ratio'], random_state=random_state))
        ]),
        "Huber Regressor": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  HuberRegressor(epsilon=model_configs['huber']['epsilon'], max_iter=model_configs['huber']['max_iter']))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=model_configs['random_forest']['n_estimators'], 
            max_depth=model_configs['random_forest']['max_depth'], 
            min_samples_leaf=model_configs['random_forest']['min_samples_leaf'],
            max_features=model_configs['random_forest']['max_features'], 
            random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=model_configs['gradient_boosting']['n_estimators'], 
            learning_rate=model_configs['gradient_boosting']['learning_rate'], 
            max_depth=model_configs['gradient_boosting']['max_depth'],
            subsample=model_configs['gradient_boosting']['subsample'], 
            random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=model_configs['xgboost']['n_estimators'], 
            learning_rate=model_configs['xgboost']['learning_rate'], 
            max_depth=model_configs['xgboost']['max_depth'],
            subsample=model_configs['xgboost']['subsample'], 
            colsample_bytree=model_configs['xgboost']['colsample_bytree'], 
            random_state=random_state,
            verbosity=0, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=model_configs['lightgbm']['n_estimators'], 
            learning_rate=model_configs['lightgbm']['learning_rate'], 
            num_leaves=model_configs['lightgbm']['num_leaves'],
            max_depth=model_configs['lightgbm']['max_depth'], 
            subsample=model_configs['lightgbm']['subsample'], 
            colsample_bytree=model_configs['lightgbm']['colsample_bytree'],
            reg_alpha=model_configs['lightgbm']['reg_alpha'], 
            reg_lambda=model_configs['lightgbm']['reg_lambda'], 
            min_child_samples=model_configs['lightgbm']['min_child_samples'],
            random_state=random_state, verbose=-1, n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=model_configs['catboost']['iterations'], 
            learning_rate=model_configs['catboost']['learning_rate'], 
            depth=model_configs['catboost']['depth'],
            random_seed=random_state, verbose=0
        )
    }


def evaluate_models_walk_forward(models: Dict[str, Any], X: np.ndarray, y_model: np.ndarray, y: np.ndarray, n_splits: int = 5, log_target: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Performs walk-forward cross-validation on all models.
    """
    logger.info(f"Starting Walk-Forward Cross-Validation (TimeSeriesSplit, N={n_splits})...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in models.items():
        logger.info(f"   Training {name}...")
        fold_metrics = {'RMSE': [], 'MAE': [], 'WAPE(%)': [], 'sMAPE(%)': [], 'MASE': []}
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_model[train_idx], y_model[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_log = model.predict(X_val)

            # Inverse transformation
            if log_target:
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_val)
                y_train_actual = np.expm1(y_tr)
            else:
                y_pred, y_true, y_train_actual = y_pred_log, y_val, y_tr

            y_pred = np.maximum(y_pred, 0)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            
            sum_actual = np.sum(np.abs(y_true))
            wape = (np.sum(np.abs(y_true - y_pred)) / sum_actual * 100 if sum_actual > 0 else np.nan)
            
            denom_smape = np.abs(y_true) + np.abs(y_pred)
            smape = (np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom_smape == 0, 1.0, denom_smape)) * 100)
            
            naive_errors = np.abs(np.diff(y_train_actual))
            naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
            mase = mae / naive_mae if (naive_mae and naive_mae > 0) else np.nan

            fold_metrics['RMSE'].append(rmse)
            fold_metrics['MAE'].append(mae)
            fold_metrics['WAPE(%)'].append(wape)
            fold_metrics['sMAPE(%)'].append(smape)
            fold_metrics['MASE'].append(mase)

        elapsed = time.time() - t0
        results[name] = {**fold_metrics, 'time_s': elapsed}
        logger.info(f"      RMSE={np.mean(fold_metrics['RMSE']):,.0f} | WAPE={np.nanmean(fold_metrics['WAPE(%)']):.1f}% | ({elapsed:.1f}s)")

    # Calculate Skill Score vs Linear Baseline
    baseline_key = "Linear Regression (Baseline)"
    if baseline_key in results:
        base_rmse = np.mean(results[baseline_key]['RMSE'])
        for name in results:
            results[name]['SS_RMSE'] = 1.0 - (np.mean(results[name]['RMSE']) / base_rmse) if base_rmse > 0 else np.nan

    return results


def save_leaderboard(results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Summarizes CV results and saves to CSV.
    """
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            'Model':       name,
            'RMSE':        f"{np.mean(res['RMSE']):,.0f} ± {np.std(res['RMSE']):,.0f}",
            'MAE':         f"{np.mean(res['MAE']):,.0f} ± {np.std(res['MAE']):,.0f}",
            'WAPE(%)':     f"{np.nanmean(res['WAPE(%)']):.1f}%",
            'sMAPE(%)':    f"{np.nanmean(res['sMAPE(%)']):.1f}%",
            'MASE':        f"{np.nanmean(res['MASE']):.3f}",
            'SS_RMSE':     f"{res['SS_RMSE']:+.3f}",
            'Train(s)':    f"{res['time_s']:.1f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Leaderboard saved to: {output_path}")


def plot_cv_metrics(results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Generates boxplots of metric variance across CV folds.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")
        rows = []
        for name, res in results.items():
            short = MODEL_SHORT_NAMES.get(name, name)
            for val in res['WAPE(%)']:
                rows.append({"Model": short, "Metric": "WAPE (%)", "Value": val})
            for val in res['sMAPE(%)']:
                rows.append({"Model": short, "Metric": "sMAPE (%)", "Value": val})
        
        plot_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["WAPE (%)", "sMAPE (%)"]):
            sns.boxplot(data=plot_df[plot_df["Metric"] == metric], x="Model", y="Value", ax=ax, palette="Blues")
            ax.set_title(metric, fontweight="bold")
            ax.tick_params(axis="x", rotation=40)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_shap_importance(model: Any, X: np.ndarray, feature_cols: List[str], output_path: Path) -> None:
    """
    Generates SHAP summary plot for the provided model.
    """
    estimator = model.named_steps['model'] if hasattr(model, 'named_steps') else model
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    X_df = pd.DataFrame(X, columns=feature_cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_df, max_display=15, show=False)
    plt.title("SHAP Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_future_predictions(model: Any, df_train: pd.DataFrame, feature_cols: List[str], pred_path: Path, output_path: Path) -> None:
    """
    Predicts revenue for the upcoming week using the best model.
    """
    if not pred_path.exists():
        logger.error(f"Prediction data not found: {pred_path}")
        return
        
    df_pred = pd.read_csv(pred_path)
    df_combined = pd.concat([df_train, df_pred], ignore_index=True).sort_values(['customer_state', 'year_week'])
    df_combined = add_dynamic_features(df_combined)
    
    df_pred_final = df_combined[df_combined['target_next_revenue'].isna()].copy()
    X_pred = df_pred_final[feature_cols].values
    
    pred_log = model.predict(X_pred)
    pred_revenue = np.maximum(np.expm1(pred_log), 0)
    
    df_result = pd.DataFrame({
        'customer_state': df_pred_final['customer_state'],
        'year_week_current': df_pred_final['year_week'],
        'predicted_next_week_revenue': pred_revenue
    }).sort_values('predicted_next_week_revenue', ascending=False)
    
    df_result.to_csv(output_path, index=False)
    logger.info(f"Future predictions saved to: {output_path}")

def run_training_pipeline(data_path: Path, pred_path: Path, reports_dir: Path, data_dir: Path, config: Dict[str, Any]) -> None:
    """
    Executes the full ML training and evaluation pipeline.
    """
    print_section_header("STARTING ML TRAINING PIPELINE")
    
    random.seed(config['random_state'])
    np.random.seed(config['random_state'])

    # 1. Prepare data
    df = load_and_prepare_data(data_path)
    df = add_dynamic_features(df)
    X, y, y_model, feature_cols = select_features(df, config['feature_selection']['exclude'], config['log_target'])
    
    # 2. Train and Evaluate
    models = get_model_definitions(config['models'], config['random_state'])
    results = evaluate_models_walk_forward(models, X, y_model, y, config['n_splits'], config['log_target'])
    
    # 3. Reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_leaderboard(results, reports_dir.parent / 'model_leaderboard.csv')
    plot_cv_metrics(results, reports_dir / 'cv_metrics_boxplot.png')
    
    # 4. SHAP for best tree model
    tree_names = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    best_tree_name = min([n for n in tree_names if n in results], key=lambda n: np.mean(results[n]['RMSE']))
    best_tree_model = models[best_tree_name]
    best_tree_model.fit(X, y_model)
    plot_shap_importance(best_tree_model, X, feature_cols, reports_dir / 'shap_feature_importance.png')
    
    # 5. Future Predictions
    best_model_name = min(results, key=lambda n: np.mean(results[n]['RMSE']))
    best_model = models[best_model_name]
    best_model.fit(X, y_model)
    make_future_predictions(best_model, df, feature_cols, pred_path, data_dir / 'predicted_next_week_revenue.csv')
    
    print_section_header("PIPELINE COMPLETED")
