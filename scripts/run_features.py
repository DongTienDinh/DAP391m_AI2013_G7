import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.features.engineering import run_feature_engineering_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    processed_dir = Config.get_path("data", "processed_olist")
    pop_file = Config.get_path("data", "external_pop")
    gdp_file = Config.get_path("data", "external_gdp")
    output_file = Config.get_path("data", "processed_olist") / "features_weekly.csv"
    pred_file = Config.get_path("data", "processed_olist") / "prediction_data.csv"
    
    run_feature_engineering_pipeline(processed_dir, pop_file, gdp_file, output_file, pred_file)

if __name__ == "__main__":
    main()
