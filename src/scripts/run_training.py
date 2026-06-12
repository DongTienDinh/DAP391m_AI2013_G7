import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.training import ModelingService
from src.orchestrator import OlistPipeline



def main():
    p = OlistPipeline()
    svc = ModelingService(
        p.config.training.model_dump(),
        p.paths.reports.figures_dir.parent,
        p.paths.data.processed_olist,
    )
    svc.run_training_pipeline(p.paths.data.processed_olist / "features_weekly.csv")


if __name__ == "__main__":
    main()
