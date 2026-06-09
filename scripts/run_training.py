from src.olist_pipeline.models.training import ModelingService
from src.olist_pipeline.pipeline import OlistPipeline


def main():
    p = OlistPipeline()
    svc = ModelingService(p.config.training.model_dump(), p.paths.reports.figures_dir.parent)
    svc.run_training_pipeline(p.paths.data.processed_olist / "features_weekly.csv")

if __name__ == "__main__":
    main()
