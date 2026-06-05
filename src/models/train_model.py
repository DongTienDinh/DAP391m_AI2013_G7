import os
import sys
import io
import time
import warnings
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

# Set global random seeds for absolute reproducibility
random.seed(42)
np.random.seed(42)

# Force UTF-8 stdout and stderr encoding for safe terminal logs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set headless mode for Matplotlib to prevent GUI errors in terminal/background execution
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

# Short display names for models
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



def get_project_paths():
    """Automatically resolve the project root directory containing 'src'"""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    for parent in current_dir.parents:
        if (parent / 'src').is_dir():
            project_root = parent
            break
    return project_root


def load_and_prepare_data(data_path):
    """Load weekly feature data, encode state codes, and sort by time"""
    print(f"\n1. LOADING DATA from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Raw data: {df.shape}")
    
    # Encode customer_state into numeric codes
    df['state_code'] = df['customer_state'].astype('category').cat.codes
    
    # Sort by year_week to ensure chronological order for TimeSeriesSplit
    df = df.sort_values('year_week').reset_index(drop=True)
    return df


def add_dynamic_features(df):
    """Calculate state-dependent dynamic features"""
    grp = df.groupby('customer_state')
    df['revenue_std_4']   = grp['revenue'].transform(lambda x: x.shift(1).rolling(4, min_periods=2).std()).fillna(0)
    df['revenue_momentum'] = (df['revenue_lag_1'] - df['revenue_lag_4']).fillna(0)
    return df


def select_features(df, log_target=True):
    """Select feature columns and apply log transformation to target to reduce skewness"""
    exclude = [
        'customer_state', 'year_week', 'target_next_revenue',
        'sales_per_capita', 'orders_per_capita',
        # Remove columns with high multicollinearity or data leakage
        'payment_value',          # corr=1.0000 with revenue
        'unique_customers',       # corr=1.0000 with order_count
        'item_count',             # corr=0.9994 with order_count
        'customers_lag_1',        # corr=1.0000 with orders_lag_1
        'purchasing_power_index', # corr=1.0000 with gdp_per_capita
        'customer_penetration',   # corr=1.0000 with penetration_gap
        'revenue_ewm_4',          # corr=0.99 with rolling features
        'revenue_ewm_8',          # corr=0.99 with rolling features
        'revenue_rolling_12',     # corr=0.99 with rolling_8
    ]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].values
    y = df['target_next_revenue'].values
    
    if log_target:
        y_model = np.log1p(y)
        print(f"   Applied log1p(target). Skewness after transformation: {pd.Series(y_model).skew():.3f}")
    else:
        y_model = y
        
    return X, y, y_model, feature_cols


def get_model_definitions(random_state=42):
    """Define 9 modeling algorithms from linear baseline to Boosting"""
    return {
        "Linear Regression (Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=10.0, random_state=random_state))
        ]),

        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state))
        ]),

        "Huber Regressor": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  HuberRegressor(epsilon=1.35, max_iter=500))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=random_state,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=random_state
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1
        ),

        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=10,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        ),

        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=0
        )
    }


def evaluate_models_walk_forward(models, X, y_model, y, log_target=True):
    """Evaluate models via Walk-Forward CV using TimeSeriesSplit.

    Metrics per fold (all computed on the original scale after ``np.expm1``):
        RMSE, MAE, WAPE, sMAPE, MASE.

    After all folds, the Skill Score (SS_RMSE) relative to the
    Linear Regression (Baseline) is appended.
    """
    print("\n3. WALK-FORWARD CROSS-VALIDATION (TimeSeriesSplit, N=5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, model in models.items():
        print(f"   Training {name}...")
        fold_rmse, fold_mae = [], []
        fold_wape, fold_smape, fold_mase = [], [], []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_model[train_idx], y_model[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_log = model.predict(X_val)

            # Inverse log-scale transformation
            if log_target:
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_val)
                y_train_actual = np.expm1(y_tr)
            else:
                y_pred = y_pred_log
                y_true = y_val
                y_train_actual = y_tr

            y_pred = np.maximum(y_pred, 0)  # Prevent negative revenue

            # --- Core metrics ---
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)

            # WAPE: sum|error| / sum|actual|
            sum_actual = np.sum(np.abs(y_true))
            wape = (np.sum(np.abs(y_true - y_pred)) / sum_actual * 100
                    if sum_actual > 0 else np.nan)

            # sMAPE: mean( 2|e| / (|a|+|p|) ) * 100
            denom_smape = np.abs(y_true) + np.abs(y_pred)
            smape = (np.mean(2.0 * np.abs(y_true - y_pred)
                             / np.where(denom_smape == 0, 1.0, denom_smape))
                     * 100)

            # MASE: MAE / in-sample naive 1-step MAE
            naive_errors = np.abs(np.diff(y_train_actual))
            naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
            mase = mae / naive_mae if (naive_mae is not np.nan and naive_mae > 0) else np.nan

            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_wape.append(wape)
            fold_smape.append(smape)
            fold_mase.append(mase)

        elapsed = time.time() - t0
        results[name] = {
            'RMSE':     fold_rmse,
            'MAE':      fold_mae,
            'WAPE(%)':  fold_wape,
            'sMAPE(%)': fold_smape,
            'MASE':     fold_mase,
            'time_s':   elapsed,
        }
        print(f"      RMSE={np.mean(fold_rmse):,.0f} ± {np.std(fold_rmse):,.0f} | "
              f"WAPE={np.nanmean(fold_wape):.1f}% | "
              f"MASE={np.nanmean(fold_mase):.3f} | "
              f"({elapsed:.1f}s)")

    # --- Compute Skill Score (SS_RMSE) vs. Baseline ---
    baseline_key = "Linear Regression (Baseline)"
    if baseline_key in results:
        baseline_rmse = np.mean(results[baseline_key]['RMSE'])
        for name in results:
            model_rmse = np.mean(results[name]['RMSE'])
            ss = 1.0 - (model_rmse / baseline_rmse) if baseline_rmse > 0 else np.nan
            results[name]['SS_RMSE'] = ss
    else:
        for name in results:
            results[name]['SS_RMSE'] = np.nan

    return results


def print_leaderboard(results):
    """Print detailed model leaderboard to Console"""
    print("\n4. MODEL LEADERBOARD (CV mean ± std):")
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            'Model':       name,
            'RMSE':        f"{np.mean(res['RMSE']):>8,.0f} ± {np.std(res['RMSE']):,.0f}",
            'MAE':         f"{np.mean(res['MAE']):>8,.0f} ± {np.std(res['MAE']):,.0f}",
            'WAPE(%)':     f"{np.nanmean(res['WAPE(%)']):>6.1f}%",
            'sMAPE(%)':    f"{np.nanmean(res['sMAPE(%)']):>6.1f}%",
            'MASE':        f"{np.nanmean(res['MASE']):.3f}",
            'SS_RMSE':     f"{res['SS_RMSE']:>+.3f}",
            'Train(s)':    f"{res['time_s']:.1f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    output_csv = get_project_paths() / "reports" / "model_leaderboard.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"\n   Saved leaderboard to -> {output_csv}")


# ---------------------------------------------------------------------------
#  Visualization functions — individual vector graphics for LaTeX
# ---------------------------------------------------------------------------

def plot_cv_metrics_boxplot(results, output_path):
    """Boxplot of WAPE and sMAPE across CV folds for every model.

    Demonstrates model stability across the 5 walk-forward splits.
    Saves as a PDF vector graphic suitable for LaTeX inclusion.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        rows = []
        for name, res in results.items():
            short = MODEL_SHORT_NAMES.get(name, name)
            for fold_idx in range(len(res['WAPE(%)'])):
                rows.append({"Model": short, "Metric": "WAPE (%)",
                             "Value": res['WAPE(%)'][fold_idx]})
                rows.append({"Model": short, "Metric": "sMAPE (%)",
                             "Value": res['sMAPE(%)'][fold_idx]})
        plot_df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

        for ax, metric in zip(axes, ["WAPE (%)", "sMAPE (%)"]):
            subset = plot_df[plot_df["Metric"] == metric]
            sns.boxplot(
                data=subset, x="Model", y="Value", ax=ax,
                palette="Blues", width=0.55, linewidth=0.8,
                flierprops=dict(marker="o", markersize=3),
            )
            ax.set_title(metric, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=40)

        fig.suptitle("Cross-Validation Metric Variance (5-Fold Walk-Forward)",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(output_path, format="png")
        plt.close(fig)
        print(f"   Saved CV metrics boxplot -> {output_path}")


def plot_relative_performance(results, output_path):
    """Grouped bar chart of MASE and Skill Score (SS_RMSE) per model.

    A horizontal dashed line at MASE = 1.0 separates models that
    outperform the naive 1-step baseline from those that do not.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        names, mase_vals, ss_vals = [], [], []
        for name, res in results.items():
            names.append(MODEL_SHORT_NAMES.get(name, name))
            mase_vals.append(np.nanmean(res['MASE']))
            ss_vals.append(res.get('SS_RMSE', np.nan))

        x = np.arange(len(names))
        bar_w = 0.35

        fig, ax1 = plt.subplots(figsize=(9, 4.5))

        # MASE bars (left y-axis)
        bars1 = ax1.bar(x - bar_w / 2, mase_vals, bar_w, label="MASE",
                        color="#4C72B0", edgecolor="white", linewidth=0.6)
        ax1.set_ylabel("MASE")
        ax1.axhline(1.0, color="#C44E52", linewidth=1.2, linestyle="--",
                    label="Naive baseline (MASE = 1)")

        # SS_RMSE bars (right y-axis)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + bar_w / 2, ss_vals, bar_w, label="SS$_{RMSE}$",
                        color="#55A868", edgecolor="white", linewidth=0.6)
        ax2.set_ylabel("Skill Score (SS$_{RMSE}$)")

        # Value annotations
        for bar, v in zip(bars1, mase_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, ss_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:+.2f}", ha="center", va="bottom", fontsize=7)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=40, ha="right")
        ax1.set_title("Relative Performance: MASE & Skill Score",
                      fontweight="bold")

        # Unified legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2,
                   loc="upper left", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(output_path, format="png")
        plt.close(fig)
        print(f"   Saved relative performance chart -> {output_path}")


def plot_shap_feature_importance(model, X, feature_cols, output_path, top_n=15):
    """SHAP summary (dot) plot for the best tree-based model.

    Uses ``shap.TreeExplainer`` and saves the top-*n* features
    as a PDF vector graphic.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")

        # Unwrap sklearn Pipeline if necessary
        estimator = (model.named_steps['model']
                     if hasattr(model, 'named_steps') else model)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)

        # Build a DataFrame so SHAP plots display readable feature names
        X_df = pd.DataFrame(X, columns=feature_cols)

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.summary_plot(
            shap_values, X_df,
            max_display=top_n,
            show=False,
            plot_size=None,  # respect the figure we created
        )
        ax = plt.gca()
        ax.set_title("SHAP Feature Importance (Top 15)", fontweight="bold")
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(output_path, format="png")
        plt.close(fig)
        print(f"   Saved SHAP summary plot -> {output_path}")


def plot_correlation_heatmap(df, feature_cols, output_path):
    """Correlation heatmap of all numeric features used in modeling.

    Computes the Pearson correlation matrix and renders an annotated
    heatmap with a diverging colormap.  Saved as a PNG graphic.
    """
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("white")

        corr = df[feature_cols].corr()

        # Mask the upper triangle for a cleaner look
        mask = np.triu(np.ones_like(corr, dtype=bool))

        n_features = len(feature_cols)
        fig_size = max(10, n_features * 0.55)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        # Annotate only when the number of features is manageable
        annotate = n_features <= 25

        sns.heatmap(
            corr,
            mask=mask,
            annot=annotate,
            fmt=".2f" if annotate else "",
            cmap="coolwarm",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            annot_kws={"size": 7},
            ax=ax,
        )

        ax.set_title("Feature Correlation Heatmap (Pearson)",
                     fontweight="bold", pad=12)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

        fig.tight_layout()
        fig.savefig(output_path, format="png")
        plt.close(fig)
        print(f"   Saved correlation heatmap -> {output_path}")


def plot_prediction_capability(model, X, y_true_log, output_path):
    """Scatter plot of Actual vs Predicted Revenue to show prediction capability."""
    with plt.rc_context(ACADEMIC_RC):
        sns.set_style("whitegrid")
        y_pred_log = model.predict(X)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)
        y_true = np.expm1(y_true_log)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, color='#4C72B0', s=10)
        
        # Plot y=x line
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Revenue')
        ax.set_ylabel('Predicted Revenue')
        ax.set_title('Prediction Capability (Actual vs Predicted)', fontweight='bold')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(output_path, format="png")
        plt.close(fig)
        print(f"   Saved prediction capability plot -> {output_path}")


def make_predictions_for_future(best_model_name, models, df_train, feature_cols, X_train, y_train_model, pred_path, output_csv_path):
    """Load prediction_data, merge dynamically to compute rolling features, make predictions and save results"""
    print(f"\n6. FUTURE PREDICTIONS USING BEST MODEL ({best_model_name})...")
    
    if not pred_path.exists():
        print(f"   [Error] Prediction data file not found: {pred_path}!")
        return
        
    df_pred = pd.read_csv(pred_path)
    
    # Merge with training set to calculate accurate rolling features
    df_combined = pd.concat([df_train, df_pred], ignore_index=True)
    df_combined = df_combined.sort_values(['customer_state', 'year_week']).reset_index(drop=True)
    
    # Recalculate dynamic features on the merged set
    df_combined['state_code'] = df_combined['customer_state'].astype('category').cat.codes
    df_combined = add_dynamic_features(df_combined)
    
    # Extract rows to predict (rows where target_next_revenue is NaN)
    df_pred_final = df_combined[df_combined['target_next_revenue'].isna()].copy()
    
    # Retrain the best model on the entire training dataset
    best_model = models[best_model_name]
    print(f"   Fitting model {best_model_name} on entire dataset...")
    best_model.fit(X_train, y_train_model)
    
    # Dự báo
    X_pred = df_pred_final[feature_cols].values
    pred_log = best_model.predict(X_pred)
    pred_revenue = np.maximum(np.expm1(pred_log), 0)
    
    df_result = pd.DataFrame({
        'customer_state': df_pred_final['customer_state'],
        'year_week_current': df_pred_final['year_week'],
        'predicted_next_week_revenue': pred_revenue
    })
    df_result = df_result.sort_values(by='predicted_next_week_revenue', ascending=False).reset_index(drop=True)
    
    print("\n   Predicted revenue for next week (Top 5 states):")
    print(df_result.head(5).to_string(index=False))
    
    df_result.to_csv(output_csv_path, index=False)
    print(f"   Saved predictions to: {output_csv_path}")


def main():
    # 1. Initialize paths
    project_root = get_project_paths()
    data_path = project_root / 'data' / 'processed' / 'olist' / 'features_weekly.csv'
    pred_path = project_root / 'data' / 'processed' / 'olist' / 'prediction_data.csv'
    output_predictions_path = project_root / 'data' / 'processed' / 'olist' / 'predicted_next_week_revenue.csv'

    reports_dir = project_root / 'reports' / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load data
    df = load_and_prepare_data(data_path)
    df = add_dynamic_features(df)

    # 3. Feature selection & log target transformation
    X, y, y_model, feature_cols = select_features(df, log_target=True)

    # 4. Define models
    models = get_model_definitions(random_state=42)

    # 5. Evaluate Walk-Forward CV
    results = evaluate_models_walk_forward(models, X, y_model, y, log_target=True)

    # 6. Print Leaderboard
    print_leaderboard(results)

    # 7. Identify best tree-based model and fit on full data for SHAP
    tree_models_list = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    best_tree_name = min(
        [n for n in tree_models_list if n in results],
        key=lambda n: np.mean(results[n]['RMSE']),
    )
    print(f"\nFitting best tree model ({best_tree_name}) on full data for SHAP...")
    best_tree_model = models[best_tree_name]
    best_tree_model.fit(X, y_model)

    # 8. Generate academic-quality PDF figures
    print("\n5. GENERATING ACADEMIC FIGURES (PDF) ...")

    plot_cv_metrics_boxplot(
        results,
        output_path=reports_dir / 'cv_metrics_boxplot.png',
    )

    plot_relative_performance(
        results,
        output_path=reports_dir / 'relative_performance.png',
    )

    plot_shap_feature_importance(
        model=best_tree_model,
        X=X,
        feature_cols=feature_cols,
        output_path=reports_dir / 'shap_feature_importance.png',
        top_n=15,
    )

    plot_correlation_heatmap(
        df=df,
        feature_cols=feature_cols,
        output_path=reports_dir / 'correlation_heatmap.png',
    )

    plot_prediction_capability(
        model=best_tree_model,
        X=X,
        y_true_log=y_model,
        output_path=reports_dir / 'predict_capability.png',
    )

    print("\nGenerating explicit SHAP for Random Forest as requested...")
    rf_model = models["Random Forest"]
    rf_model.fit(X, y_model)
    plot_shap_feature_importance(
        model=rf_model,
        X=X,
        feature_cols=feature_cols,
        output_path=reports_dir / 'shap_rf.png',
        top_n=15,
    )

    # 9. Find best model to make future predictions
    best_model_name = min(results, key=lambda n: np.mean(results[n]['RMSE']))
    make_predictions_for_future(best_model_name, models, df, feature_cols, X, y_model, pred_path, output_predictions_path)

    print("\nML pipeline executed successfully!")


if __name__ == "__main__":
    main()
