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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.linear_model     import LinearRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import TimeSeriesSplit
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score
from xgboost                  import XGBRegressor
import lightgbm as lgb
from catboost                 import CatBoostRegressor

warnings.filterwarnings('ignore')

# Color theme and display constants
C = {
    "bg":      "#F7F9FC",
    "panel":   "#FFFFFF",
    "border":  "#E2E8F0",
    "text":    "#1A202C",
    "sub":     "#718096",
    "grid":    "#EDF2F7",
    "best":    "#2B6CB0",
    "worst":   "#FC8181",
}

_RAW_COLORS = [
    "#A0AEC0", "#CBD5E0", "#B2BFCC", "#90CDF4",
    "#48BB78", "#38A169", "#2F855A", "#276749", "#1C4532"
]


def _make_bar_colors(highlight_idx, worst_idx=None):
    """Return list of colors, highlighting the best / worst models."""
    cols = list(_RAW_COLORS)
    if worst_idx is not None:
        cols[worst_idx] = C["worst"]
    cols[highlight_idx] = C["best"]
    return cols


def _bar_labels(ax, bars, values, fmt, offset_frac=0.015, fs=7.5):
    """Draw numeric labels on top of each bar."""
    ylim = ax.get_ylim()
    dy   = (ylim[1] - ylim[0]) * offset_frac
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            fmt(v), ha="center", va="bottom",
            fontsize=fs, fontweight="bold", color=C["text"], zorder=5,
        )


def _rank_asc(vals):
    """Rank ascending (smaller = better rank = smaller rank value)."""
    order = np.argsort(vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(vals) + 1)
    return ranks


def _rank_desc(vals):
    return _rank_asc([-v for v in vals])



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
    """Evaluate models via Walk-Forward CV using TimeSeriesSplit"""
    print("\n3. WALK-FORWARD CROSS-VALIDATION (TimeSeriesSplit, N=5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        fold_rmse, fold_mae, fold_r2, fold_mape = [], [], [], []
        t0 = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_model[train_idx], y_model[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred_log = model.predict(X_val)
            
            if log_target:
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_val)
            else:
                y_pred = y_pred_log
                y_true = y_val
                
            y_pred = np.maximum(y_pred, 0) # Prevent negative revenue
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            r2   = r2_score(y_true, y_pred)
            
            mask = y_true > 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
            
            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_r2.append(r2)
            fold_mape.append(mape)
            
        elapsed = time.time() - t0
        results[name] = {
            'RMSE':    fold_rmse,
            'MAE':     fold_mae,
            'R2':      fold_r2,
            'MAPE(%)': fold_mape,
            'time_s':  elapsed
        }
        print(f"      RMSE={np.mean(fold_rmse):,.0f} ± {np.std(fold_rmse):,.0f} | "
              f"R²={np.mean(fold_r2):.3f} | MAPE={np.nanmean(fold_mape):.1f}% | "
              f"({elapsed:.1f}s)")
              
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
            'R²':          f"{np.mean(res['R2']):.4f}",
            'MAPE(%)':     f"{np.nanmean(res['MAPE(%)']):>6.1f}%",
            'Train(s)':    f"{res['time_s']:.1f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))


def plot_and_save_dashboard(results, fi_models, feature_cols, output_path, project_root):
    """Generate model comparison dashboard and save as image file"""
    print(f"\n5. GENERATING EVALUATION DASHBOARD -> {output_path}...")
    
    name_mapping = {
        "Linear Regression (Baseline)": "Linear",
        "Ridge Regression": "Ridge",
        "ElasticNet": "ElasticNet",
        "Huber Regressor": "Huber",
        "Random Forest": "RF",
        "Gradient Boosting": "GBR",
        "XGBoost": "XGBoost",
        "LightGBM": "LGBM",
        "CatBoost": "CatBoost"
    }
    
    MODEL_NAMES = [name_mapping[k] for k in results.keys()]
    RMSE_MEANS = [np.mean(results[k]['RMSE']) for k in results.keys()]
    RMSE_STDS  = [np.std(results[k]['RMSE']) for k in results.keys()]
    R2_MEANS   = [np.mean(results[k]['R2']) for k in results.keys()]
    R2_STDS    = [np.std(results[k]['R2']) for k in results.keys()]
    MAPE_MEANS = [np.nanmean(results[k]['MAPE(%)']) for k in results.keys()]
    
    FOLD_RMSE = {name_mapping[k]: results[k]['RMSE'] for k in results.keys()}
    
    # Automatically define the best / worst models
    BEST_MODEL  = name_mapping[min(results, key=lambda n: np.mean(results[n]['RMSE']))]
    WORST_MODEL = name_mapping[max(results, key=lambda n: np.mean(results[n]['RMSE']))]
    
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    BEST_FI_MODEL = min([n for n in tree_models if n in results], key=lambda n: np.mean(results[n]['RMSE']))
    
    # Feature Importance of the best tree model
    m = fi_models[BEST_FI_MODEL]
    if hasattr(m, 'feature_importances_'):
        importances = m.feature_importances_
    else:
        importances = m.named_steps['model'].feature_importances_
        
    imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    FEATURE_NAMES = list(imp.index[:15])
    FEATURE_IMP = list(imp.values[:15])
    
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   C["panel"],
        "figure.facecolor": C["bg"],
        "axes.edgecolor":   C["border"],
        "axes.labelcolor":  C["text"],
        "xtick.color":      C["sub"],
        "ytick.color":      C["sub"],
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       C["grid"],
        "grid.linewidth":   0.8,
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
        "axes.titlecolor":  C["text"],
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
    })
    
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(C["bg"])
    
    fig.text(0.5, 0.975, "Model Selection — Weekly Revenue Forecasting",
             ha="center", va="top", fontsize=18, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.950, "Compare 9 models via 5-Fold Cross-Validation  |  Revenue Unit: VND",
             ha="center", va="top", fontsize=10, color=C["sub"])
             
    gs = gridspec.GridSpec(2, 3, figure=fig, top=0.91, bottom=0.07, hspace=0.52, wspace=0.38, left=0.07, right=0.97)
    BAR_KW = dict(edgecolor="white", linewidth=1.5, zorder=3)
    BEST_IDX  = MODEL_NAMES.index(BEST_MODEL)
    WORST_IDX = MODEL_NAMES.index(WORST_MODEL)
    N_FOLDS   = len(next(iter(FOLD_RMSE.values())))
    

            
    # ① Panel A: CV RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    cols_a = _make_bar_colors(BEST_IDX, WORST_IDX)
    bars_a = ax1.bar(MODEL_NAMES, RMSE_MEANS, color=cols_a, **BAR_KW)
    ax1.errorbar(range(len(MODEL_NAMES)), RMSE_MEANS, yerr=RMSE_STDS, fmt="none", ecolor="#4A5568", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
    ax1.set_title("① CV RMSE  (lower = better)")
    ax1.set_ylabel("RMSE (VND)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax1.set_xticks(range(len(MODEL_NAMES)))
    ax1.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax1.set_ylim(0, max(RMSE_MEANS) * 1.25)
    _bar_labels(ax1, bars_a, RMSE_MEANS, fmt=lambda v: f"{v/1000:.1f}K")
    ax1.annotate("Best", xy=(BEST_IDX, RMSE_MEANS[BEST_IDX]), xytext=(BEST_IDX + 0.6, RMSE_MEANS[BEST_IDX] + max(RMSE_STDS) * 0.35),
                 fontsize=8, color=C["best"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["best"], lw=1.4))
                 
    # ② Panel B: CV R²
    ax2 = fig.add_subplot(gs[0, 1])
    cols_b = _make_bar_colors(BEST_IDX, WORST_IDX)
    for idx, r2_val in enumerate(R2_MEANS):
        if r2_val < 0:
            cols_b[idx] = "#FEB2B2" if r2_val > -1.0 else "#FC8181"
            
    bars_b = ax2.bar(MODEL_NAMES, R2_MEANS, color=cols_b, **BAR_KW)
    ax2.errorbar(range(len(MODEL_NAMES)), R2_MEANS, yerr=R2_STDS, fmt="none", ecolor="#4A5568", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
    ax2.axhline(0, color="#4A5568", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_title("② CV R²  (higher = better)")
    ax2.set_ylabel("R² Score")
    ax2.set_xticks(range(len(MODEL_NAMES)))
    ax2.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    for i in range(len(MODEL_NAMES)):
        v = R2_MEANS[i]
        if v > 0:
            bar = bars_b[i]
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["text"])
            
    # ③ Panel C: CV MAPE
    ax3 = fig.add_subplot(gs[0, 2])
    cols_c = _make_bar_colors(MAPE_MEANS.index(min(MAPE_MEANS)), WORST_IDX)
    bars_c = ax3.bar(MODEL_NAMES, MAPE_MEANS, color=cols_c, **BAR_KW)
    ax3.set_title("③ CV MAPE %  (lower = better)")
    ax3.set_ylabel("MAPE (%)")
    ax3.set_xticks(range(len(MODEL_NAMES)))
    ax3.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax3.set_ylim(0, max(MAPE_MEANS) * 1.18)
    _bar_labels(ax3, bars_c, MAPE_MEANS, fmt=lambda v: f"{v:.1f}%")
    
    # Inset zoom
    inset = ax3.inset_axes([0.40, 0.32, 0.58, 0.58])
    tree_m = MODEL_NAMES[4:]
    tree_v = MAPE_MEANS[4:]
    tree_c = cols_c[4:]
    b_in   = inset.bar(range(len(tree_m)), tree_v, color=tree_c, edgecolor="white", linewidth=1.2)
    inset.set_xticks(range(len(tree_m)))
    inset.set_xticklabels(tree_m, fontsize=6.5, rotation=35, ha="right")
    inset.set_ylim(min(tree_v) - 2, max(tree_v) + 4)
    inset.set_title("Zoom: Tree Models", fontsize=7, pad=2)
    inset.tick_params(axis="y", labelsize=6.5)
    inset.set_facecolor(C["bg"])
    inset.grid(color=C["grid"], linewidth=0.6)
    for b, v in zip(b_in, tree_v):
        inset.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")
        
    # ④ Panel D: Stability (RMSE per fold, log scale)
    ax4 = fig.add_subplot(gs[1, 0])
    folds = list(range(1, N_FOLDS + 1))
    for i, name in enumerate(MODEL_NAMES):
        is_best  = name == BEST_MODEL
        is_worst = name == WORST_MODEL
        lw    = 3.0 if is_best else (2.0 if is_worst else 1.4)
        alpha = 1.0 if (is_best or is_worst) else 0.55
        ms    = 6   if (is_best or is_worst) else 4
        ax4.plot(folds, FOLD_RMSE[name], marker="o", color=_RAW_COLORS[i] if not is_worst else C["worst"], linewidth=lw, alpha=alpha, markersize=ms, label=name, zorder=3 if (is_best or is_worst) else 2)
    ax4.set_yscale("log")
    ax4.set_title("④ RMSE per Fold — Stability (Log Scale)")
    ax4.set_xlabel("CV Fold")
    ax4.set_ylabel("RMSE (log)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax4.set_xticks(folds)
    ax4.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9, edgecolor=C["border"])
    
    # ⑤ Panel E: Leaderboard
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    rank_r = _rank_asc(RMSE_MEANS)
    rank_q = _rank_desc(R2_MEANS)
    rank_m = _rank_asc(MAPE_MEANS)
    overall = rank_r + rank_q + rank_m
    sorted_idx = np.argsort(overall)
    
    ax5.text(0.5, 0.97, "⑤ Overall Leaderboard", transform=ax5.transAxes, ha="center", va="top", fontsize=11, fontweight="bold", color=C["text"])
    ax5.text(0.5, 0.89, "Total Rank = RMSE Rank + R2 Rank + MAPE Rank", transform=ax5.transAxes, ha="center", va="top", fontsize=8, color=C["sub"])
    
    COLS_X  = [0.04, 0.14, 0.42, 0.57, 0.72, 0.89]
    COL_HDR = ["#",  "Model", "RMSE",  "R2",   "MAPE",  "Score"]
    ROW_TOP = 0.80
    ROW_H   = 0.082
    MEDALS  = {0: "1st", 1: "2nd", 2: "3rd"}
    ROW_BG  = {0: "#EBF8FF", 1: "#F0FFF4", 2: "#FFFAF0"}
    
    for cx, lbl in zip(COLS_X, COL_HDR):
        ax5.text(cx, ROW_TOP + 0.01, lbl, transform=ax5.transAxes, fontsize=8.5, fontweight="bold", color=C["text"], va="top")
    ax5.plot([0.02, 0.98], [ROW_TOP - 0.005]*2, transform=ax5.transAxes, color=C["border"], linewidth=1, zorder=2)
    
    for pos, idx in enumerate(sorted_idx):
        y      = ROW_TOP - (pos + 1) * ROW_H
        bg     = ROW_BG.get(pos, C["panel"])
        prefix = MEDALS.get(pos, str(pos + 1))
        rect = FancyBboxPatch((0.02, y - 0.005), 0.96, ROW_H - 0.006, transform=ax5.transAxes, boxstyle="round,pad=0.005", linewidth=0.5, edgecolor=C["border"], facecolor=bg, zorder=1)
        ax5.add_patch(rect)
        
        row_vals = [
            prefix,
            MODEL_NAMES[idx],
            f"{RMSE_MEANS[idx]/1000:.1f}K",
            f"{R2_MEANS[idx]:.3f}",
            f"{MAPE_MEANS[idx]:.1f}%",
            str(int(overall[idx])),
        ]
        color = C["best"] if pos == 0 else C["text"]
        fw    = "bold"   if pos < 3   else "normal"
        for cx, val in zip(COLS_X, row_vals):
            ax5.text(cx, y + ROW_H * 0.35, val, transform=ax5.transAxes, fontsize=8, fontweight=fw, color=color, va="center")
            
    # ⑥ Panel F: Feature Importance
    ax6 = fig.add_subplot(gs[1, 2])
    n_fi      = len(FEATURE_NAMES)
    cmap_vals = np.linspace(0.25, 0.90, n_fi)
    fi_colors = plt.cm.Blues(cmap_vals)
    h_bars = ax6.barh(list(range(n_fi)), FEATURE_IMP[::-1], color=fi_colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax6.set_yticks(list(range(n_fi)))
    ax6.set_yticklabels(FEATURE_NAMES[::-1], fontsize=8)
    ax6.set_title(f"⑥ Feature Importance — {BEST_FI_MODEL} (Top {n_fi})")
    ax6.set_xlabel("Importance Score")
    ax6.set_xlim(0, max(FEATURE_IMP) * 1.28)
    for bar, v in zip(h_bars, FEATURE_IMP[::-1]):
        ax6.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5, fontweight="bold", color=C["text"])
    ax6.annotate(f"Top feature:\\n{FEATURE_NAMES[0]}", xy=(FEATURE_IMP[0], n_fi - 1), xytext=(0.17, n_fi - 1),
                 fontsize=8, color=C["best"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["best"], lw=1.2))
                 
    best_i = MODEL_NAMES.index(BEST_MODEL)
    footer = (
        f">> Mo hinh duoc chon: {BEST_FI_MODEL}  |  "
        f"RMSE: {RMSE_MEANS[best_i]/1000:.1f}K  |  "
        f"R2: {R2_MEANS[best_i]:.3f}  |  "
        f"MAPE: {MAPE_MEANS[best_i]:.1f}%  |  "
        f"On dinh nhat qua cac fold"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9, color=C["best"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor="#BEE3F8", linewidth=1.2))
             
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    print(f"   Đã lưu Dashboard vào: {output_path}")


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
    dashboard_path = project_root / 'reports' / 'model_comparison.png'
    output_predictions_path = project_root / 'data' / 'processed' / 'olist' / 'predicted_next_week_revenue.csv'
    
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
    
    # 7. Train on entire dataset to get Feature Importance
    fi_models = {}
    tree_models_list = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    print("\nFitting tree models for Feature Importance analysis...")
    for name in tree_models_list:
        if name in models:
            m = models[name]
            m.fit(X, y_model)
            fi_models[name] = m
            
    # 8. Plot comparison Dashboard
    plot_and_save_dashboard(results, fi_models, feature_cols, dashboard_path, project_root)
    
    # 9. Find best model to make future predictions
    best_model_name = min(results, key=lambda n: np.mean(results[n]['RMSE']))
    make_predictions_for_future(best_model_name, models, df, feature_cols, X, y_model, pred_path, output_predictions_path)
    
    print("\nML pipeline executed successfully!")


if __name__ == "__main__":
    main()
