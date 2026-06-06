import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.models.training import run_training_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    data_path = Config.get_path("data", "processed_olist") / "features_weekly.csv"
    pred_path = Config.get_path("data", "processed_olist") / "prediction_data.csv"
    reports_dir = Config.get_path("reports", "figures_dir")
    data_dir = Config.get_path("data", "processed_olist")
    
    run_training_pipeline(data_path, pred_path, reports_dir, data_dir, Config.training)

if __name__ == "__main__":
    main()
