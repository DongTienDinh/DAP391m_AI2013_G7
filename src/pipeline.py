import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import OlistPipeline  # noqa: F401 — backward compat for run_*.py


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_subheader(title: str) -> None:
    print(f"\n  ▶ {title}")
    print("  " + "-" * (len(title) + 4))


def _show_file(path: Path, head: int = 5) -> None:
    if path.exists():
        import pandas as pd
        df = pd.read_csv(path)
        print(f"     File: {path}")
        print(f"     Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"     Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
        print(f"     Preview (top {head}):")
        print(df.head(head).to_string(index=False))
    else:
        print(f"     (file not found: {path})")


def _show_leaderboard(path: Path) -> None:
    if not path.exists():
        print(f"     (file not found: {path})")
        return
    import pandas as pd
    df = pd.read_csv(path)
    print(f"     File: {path}")
    print(f"     Models: {len(df)}")
    for _, r in df.iterrows():
        if r["model"] == df.iloc[0]["model"]:
            tag = "  🥇"
        elif r["model"] == df.iloc[1]["model"]:
            tag = "  🥈"
        elif r["model"] == df.iloc[2]["model"]:
            tag = "  🥉"
        else:
            tag = "    "
        print(f"     {tag} {r['model']:35s}  RMSE={r['RMSE']:>10,.2f}  MAE={r['MAE']:>10,.2f}  SS={r.get('SS_RMSE', 0):>+.3f}")


def _show_top_states(path: Path, n: int = 10) -> None:
    if not path.exists():
        print(f"     (file not found: {path})")
        return
    import pandas as pd
    df = pd.read_csv(path)
    cols = ["EPS_rank", "customer_state", "EPS_score", "tier", "dominant_component", "state_display"]
    cols = [c for c in cols if c in df.columns]
    print(f"     Top {n} states by EPS:")
    print(df.head(n)[cols].to_string(index=False))


def _show_weights(path: Path) -> None:
    if not path.exists():
        print(f"     (file not found: {path})")
        return
    import json
    with open(path) as f:
        cfg = json.load(f)
    w = cfg.get("w_star", {})
    print(f"     Optimal weights (gamma={cfg.get('gamma', '?')}):")
    for c, v in w.items():
        print(f"       w({c:3s}) = {v:.4f}")


def _show_ranking_comparison(path: Path) -> None:
    if not path.exists():
        print(f"     (file not found: {path})")
        return
    with open(path) as f:
        print(f.read())


def _show_predicted(path: Path, head: int = 5) -> None:
    if not path.exists():
        print(f"     (file not found: {path})")
        return
    import pandas as pd
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if "predicted" in c]
    if pred_col:
        print(f"     Top {head} predicted states:")
        print(df.head(head).to_string(index=False))


def _run_step(number: int, label: str, desc: str, main_fn, results_fn=None) -> None:
    _print_header(f"Step {number}: {label}")
    print(f"  {desc}")
    ans = input("\n  Run this step? (y/n) [y]: ").strip().lower()
    if ans == "n":
        print("  ⏭ Skipped.")
        return
    print("  Running...")
    main_fn()
    print("  ✅ Done.")
    if results_fn:
        results_fn()


def run_all() -> None:
    """Run the complete pipeline non-interactively."""
    from src.scripts.run_cleaning import main as clean_main
    from src.scripts.run_features import main as feat_main
    from src.scripts.run_training import main as train_main
    from src.scripts.run_scoring import main as score_main
    from src.scripts.run_xai import main as xai_main
    from src.scripts.run_ranking_comparison import main as rc_main

    clean_main()
    feat_main()
    train_main()
    score_main()
    xai_main()
    rc_main()


def run_interactive() -> None:
    """Interactive step-by-step pipeline."""
    from src.core.config import get_config
    cfg = get_config()
    P = cfg.paths

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  COMPASS-XAI: Interactive Pipeline (DAP391m 10-Step Framework)  ║")
    print("╚" + "═" * 68 + "╝")

    # ── Part 1: Problem & Data Understanding ──────────────────────────────
    _print_header("PART 1 — Problem & Data Understanding (Steps 1-5)")
    print("  This section covers business understanding, data collection,")
    print("  schema validation, and cleaning of the Olist e-commerce dataset.")
    print()

    ans = input("  Run Part 1 (Data Cleaning)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_cleaning import main as clean_main
        _print_subheader("Running: Data Collection & Cleaning")
        clean_main()
        _print_subheader("Results: Data Cleaning")
        _show_file(P.data.processed_olist / "orders.csv", head=3)
        print(f"     → 8 cleaned files saved to {P.data.processed_olist}")
    else:
        print("  ⏭ Skipped.")

    # ── Part 2: Feature Engineering ────────────────────────────────────────
    _print_header("PART 2 — Feature Engineering (Step 6)")
    print("  Engineering 35+ features: seasonality, lags, rolling windows,")
    print("  IBGE demographics, growth rates, and penetration metrics.")
    print()

    ans = input("  Run Part 2 (Feature Engineering)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_features import main as feat_main
        _print_subheader("Running: Feature Engineering")
        feat_main()
        _print_subheader("Results: Feature Engineering")
        _show_file(P.data.processed_olist / "features_weekly.csv", head=3)
        _show_file(P.data.processed_olist / "prediction_data.csv", head=3)
    else:
        print("  ⏭ Skipped.")

    # ── Part 3: Modeling & Evaluation ──────────────────────────────────────
    _print_header("PART 3 — Modeling, Evaluation & Scoring (Steps 7-8-9)")
    print("  Walk-forward CV of 9 models (Ridge → CatBoost), champion")
    print("  selection, EPS scoring with SLSQP entropy optimization.")
    print()

    ans = input("  Run Part 3a (Model Training)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_training import main as train_main
        _print_subheader("Running: Model Training & Evaluation")
        train_main()
        _print_subheader("Results: Model Leaderboard")
        _show_leaderboard(P.reports.leaderboard)
        _show_predicted(P.data.processed_olist / "predicted_next_week_revenue.csv")
    else:
        print("  ⏭ Skipped.")

    ans = input("  Run Part 3b (EPS Scoring)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_scoring import main as score_main
        _print_subheader("Running: EPS Scoring & Ranking")
        score_main()
        _print_subheader("Results: EPS Rankings")
        _show_weights(P.outputs.w_star)
        _show_top_states(P.outputs.eps_results, n=10)
    else:
        print("  ⏭ Skipped.")

    ans = input("  Run Part 3c (Ranking Comparison)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_ranking_comparison import main as rc_main
        _print_subheader("Running: EPS vs Baseline Comparison")
        rc_main()
        _print_subheader("Results: Ranking Comparison")
        _show_ranking_comparison(P.reports.figures_dir.parent / "ranking_comparison_summary.txt")
    else:
        print("  ⏭ Skipped.")

    # ── Part 4: Conclusion & XAI ───────────────────────────────────────────
    _print_header("PART 4 — Conclusion & AI Reflection (Step 10)")
    print("  Generating XAI narratives (Gemini API or rule-based fallback),")
    print("  SHAP alignment, and explanation reports.")
    print()

    ans = input("  Run Part 4 (XAI Narratives)? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_xai import main as xai_main
        _print_subheader("Running: XAI Narrative Generation")
        xai_main()
        _print_subheader("Results: XAI Reports")
        _show_file(P.outputs.xai_report_csv, head=5)
    else:
        print("  ⏭ Skipped.")

    # ── Done ───────────────────────────────────────────────────────────────
    _print_header("PIPELINE COMPLETED")
    print("  All selected stages finished. You can now launch the dashboard:")
    print()
    print("    streamlit run app/streamlit_app.py")
    print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
