from src.olist_pipeline.features.engineering import FeatureEngineeringService
from src.olist_pipeline.pipeline import OlistPipeline


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
