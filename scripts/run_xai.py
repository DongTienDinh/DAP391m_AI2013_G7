import sys
import os
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.olist_pipeline.analysis.xai import run_xai_pipeline
from src.olist_pipeline.utils.config_loader import Config

def main():
    eps_path = Config.get_path("outputs", "eps_results")
    w_star_path = Config.get_path("outputs", "w_star")
    shap_profiles_path = Config.get_path("outputs", "shap_profiles")
    output_dir = Config.get_path("outputs", "eps_dir")
    api_key = os.environ.get("GEMINI_API_KEY")
    
    run_xai_pipeline(eps_path, w_star_path, shap_profiles_path, output_dir, api_key, Config.inference)

if __name__ == "__main__":
    main()
