import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.data.cleaning import run_cleaning_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    raw_dir = Config.get_path("data", "raw_olist")
    processed_dir = Config.get_path("data", "processed_olist")
    run_cleaning_pipeline(raw_dir, processed_dir)

if __name__ == "__main__":
    main()
