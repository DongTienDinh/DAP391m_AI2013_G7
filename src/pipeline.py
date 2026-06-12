import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import OlistPipeline  # noqa: F401


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _trunc(text: str, n: int = 60) -> str:
    return str(text)[:n] + "..." if len(str(text)) > n else str(text)


def _show_file(path: Path, head: int = 3, skip_cols: list[str] | None = None) -> None:
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    skip = skip_cols or []
    show_cols = [c for c in df.columns if c not in skip][:5]
    print(f"     [{df.shape[0]} rows x {df.shape[1]} cols]")
    for _, r in df.head(head).iterrows():
        for c in show_cols:
            print(f"       {c}: {_trunc(r[c], 60)}")
        print()


def _show_leaderboard(path: Path) -> None:
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    print(f"     {'Rank':>4s}  {'Model':<30s}  {'RMSE':>12s}  {'MAE':>10s}  {'Skill':>8s}")
    print(f"     {'----':>4s}  {'----':<30s}  {'----':>12s}  {'----':>10s}  {'----':>8s}")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        ss = r.get("SS_RMSE", 0)
        rank = f"#{i}" if i <= 3 else "   "
        print(f"     {rank:>4s}  {r['model']:<30s}  {r['RMSE']:>12,.2f}  {r['MAE']:>10,.2f}  {ss:>+8.3f}")


def _show_top_states(path: Path, n: int = 10) -> None:
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    cols = ["EPS_rank", "customer_state", "EPS_score", "tier", "dominant_component"]
    cols = [c for c in cols if c in df.columns]
    print(f"     {'Rank':>4s}  {'State':>6s}  {'Score':>6s}  {'Tier':<6s}  {'Driver':<10s}")
    print(f"     {'----':>4s}  {'----':>6s}  {'----':>6s}  {'----':<6s}  {'------':<10s}")
    for _, r in df.head(n).iterrows():
        print(f"     {int(r['EPS_rank']):>4d}  {r['customer_state']:>6s}  {r['EPS_score']:>6.1f}  {r.get('tier',''):<6s}  {r.get('dominant_component',''):<10s}")


def _show_weights(path: Path) -> None:
    if not path.exists():
        return
    import json
    with open(path) as f:
        cfg = json.load(f)
    w = cfg.get("w_star", {})
    print(f"     gamma={cfg.get('gamma', '?')}")
    for c, v in w.items():
        print(f"       w({c:3s}) = {v:.4f}")


def _show_ranking_comparison(path: Path) -> None:
    if not path.exists():
        return
    print(f"     (see full report: {path})")
    for line in open(path):
        line = line.rstrip()
        if any(k in line for k in ["Spearman", "Top-5 overlap", "Max rank", "Mean rank"]):
            print(f"     {line}")


def _show_predicted(path: Path, head: int = 5) -> None:
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if "predicted" in c]
    if not pred_col:
        return
    print(f"     Top {head} predicted states:")
    for _, r in df.head(head).iterrows():
        print(f"       {r['customer_state']}: {r[pred_col[0]]:.2f}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_all() -> None:
    from src.scripts.run_cleaning import main as clean_main
    from src.scripts.run_features import main as feat_main
    from src.scripts.run_training import main as train_main
    from src.scripts.run_scoring import main as score_main
    from src.scripts.run_xai import main as xai_main
    from src.scripts.run_ranking_comparison import main as rc_main
    from src.scripts.run_figures import main as fig_main
    from src.scripts.download_geojson import download_geojson

    download_geojson()
    clean_main()
    feat_main()
    train_main()
    score_main()
    rc_main()
    fig_main()
    xai_main()


def run_interactive() -> None:
    from src.core.config import get_config
    cfg = get_config()
    P = cfg.paths

    print()
    print("=" * 70)
    print("  COMPASS-XAI: Interactive Pipeline (DAP391m 10-Step Framework)")
    print("=" * 70)

    # Step 0
    _print_header("Step 0: Download Brazil GeoJSON")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.download_geojson import download_geojson
        download_geojson()
    else:
        print("  -- Skipped.")

    # Part 1
    _print_header("PART 1 -- Data Collection & Cleaning (Steps 1-5)")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_cleaning import main as clean_main
        clean_main()
        print("  -> Results:")
        print(f"     8 tables saved to {P.data.processed_olist}")
    else:
        print("  -- Skipped.")

    # Part 2
    _print_header("PART 2 -- Feature Engineering (Step 6)")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_features import main as feat_main
        feat_main()
    else:
        print("  -- Skipped.")

    # Part 3a
    _print_header("PART 3a -- Model Training (Steps 7-8)")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_training import main as train_main
        train_main()
        print("  -> Leaderboard:")
        _show_leaderboard(P.reports.leaderboard)
        _show_predicted(P.data.processed_olist / "predicted_next_week_revenue.csv")
    else:
        print("  -- Skipped.")

    # Part 3b
    _print_header("PART 3b -- EPS Scoring (Step 9)")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_scoring import main as score_main
        score_main()
        print("  -> Rankings:")
        _show_weights(P.outputs.w_star)
        _show_top_states(P.outputs.eps_results, n=10)
    else:
        print("  -- Skipped.")

    # Part 3c
    _print_header("PART 3c -- Ranking Comparison")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_ranking_comparison import main as rc_main
        rc_main()
        _show_ranking_comparison(P.reports.figures_dir.parent / "ranking_comparison_summary.txt")
    else:
        print("  -- Skipped.")

    # Part 3d
    _print_header("PART 3d -- Generate Figures")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_figures import main as fig_main
        fig_main()
    else:
        print("  -- Skipped.")

    # Part 4
    _print_header("PART 4 -- XAI Narratives (Step 10)")
    ans = input("  Run? (y/n) [y]: ").strip().lower()
    if ans != "n":
        from src.scripts.run_xai import main as xai_main
        xai_main()
        print("  -> Preview:")
        if P.outputs.xai_report_csv.exists():
            import pandas as pd
            df = pd.read_csv(P.outputs.xai_report_csv)
            print(f"     [{len(df)} states]")
            for _, r in df.head(5).iterrows():
                brief = _trunc(r.get("brief", r.get("narrative_brief", "")), 70)
                print(f"       #{int(r['rank'])} {r['state']}: {brief}")
    else:
        print("  -- Skipped.")

    _print_header("PIPELINE COMPLETED")
    print("  To launch the dashboard:")
    print("    streamlit run app/streamlit_app.py")
    print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
