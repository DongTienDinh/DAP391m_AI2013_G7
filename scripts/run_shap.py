import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.analysis.shap import run_shap_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    data_path = Config.get_path("data", "processed_olist") / "features_weekly.csv"
    eps_path = Config.get_path("outputs", "eps_results")
    w_star_path = Config.get_path("outputs", "w_star")
    output_dir = Config.get_path("outputs", "shap_dir")
    
    run_shap_pipeline(data_path, eps_path, w_star_path, output_dir, Config.training)

if __name__ == "__main__":
    main()
