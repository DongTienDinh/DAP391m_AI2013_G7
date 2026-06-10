import os

from dotenv import load_dotenv

from src.olist_pipeline.analysis.providers.gemini import GeminiProvider
from src.olist_pipeline.analysis.scoring import ScoringService
from src.olist_pipeline.analysis.xai import XAIService
from src.olist_pipeline.core.config import get_config
from src.olist_pipeline.core.logging_setup import setup_logger
from src.olist_pipeline.data.cleaning import DataService
from src.olist_pipeline.features.engineering import FeatureEngineeringService
from src.olist_pipeline.models.training import ModelingService

logger = setup_logger("pipeline_orchestrator")


class OlistPipeline:
    """E2E Orchestrator for the Olist Expansion Pipeline."""

    def __init__(self):
        load_dotenv()
        self.config = get_config()
        self.paths = self.config.paths

    def run_full_pipeline(self) -> None:
        """Executes all stages from ingestion to XAI."""
        logger.info("Starting Full E2E Pipeline Execution...")

        # 1. Data Ingestion & Cleaning
        data_svc = DataService(self.paths.data.raw_olist, self.paths.data.processed_olist)
        data_svc.run_cleaning_pipeline()

        # 2. Feature Engineering
        feat_svc = FeatureEngineeringService(
            self.paths.data.processed_olist,
            self.paths.data.processed_olist / "features_weekly.csv",
            self.paths.data.processed_olist / "prediction_data.csv",
        )
        feat_svc.run_feature_pipeline(self.paths.data.external_pop, self.paths.data.external_gdp)

        # 3. Model Training & Benchmarking
        model_svc = ModelingService(
            self.config.training.model_dump(), self.paths.reports.figures_dir.parent
        )
        model_svc.run_training_pipeline(self.paths.data.processed_olist / "features_weekly.csv")

        # 4. EPS Scoring
        score_svc = ScoringService(self.config.inference.model_dump(), self.paths.outputs.eps_dir)
        score_svc.run_scoring_pipeline(
            self.paths.data.processed_olist / "features_weekly.csv",
            self.paths.data.processed_olist / "prediction_data.csv",
        )

        # 5. XAI
        llm = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY", ""))
        xai_svc = XAIService(self.paths.outputs.eps_dir, llm_provider=llm)
        xai_svc.run_xai_pipeline(self.paths.outputs.eps_results, self.paths.outputs.w_star)

        logger.info("Full E2E Pipeline Execution completed successfully.")


if __name__ == "__main__":
    OlistPipeline().run_full_pipeline()
