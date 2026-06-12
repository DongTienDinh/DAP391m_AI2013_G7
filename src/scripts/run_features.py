import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineering import FeatureEngineeringService
from src.orchestrator import OlistPipeline



def main():
    p = OlistPipeline()
    svc = FeatureEngineeringService(
        p.paths.data.processed_olist,
        p.paths.data.processed_olist / "features_weekly.csv",
        p.paths.data.processed_olist / "prediction_data.csv"
    )
    svc.run_feature_pipeline(p.paths.data.external_pop, p.paths.data.external_gdp)

if __name__ == "__main__":
    main()
