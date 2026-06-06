import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.analysis.scoring import run_scoring_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    features_path = Config.get_path("data", "processed_olist") / "features_weekly.csv"
    pred_path = Config.get_path("data", "processed_olist") / "predicted_next_week_revenue.csv"
    output_dir = Config.get_path("outputs", "eps_dir")
    fig_dir = Config.get_path("reports", "figures_dir")
    
    run_scoring_pipeline(features_path, pred_path, output_dir, fig_dir, Config.inference)

if __name__ == "__main__":
    main()
