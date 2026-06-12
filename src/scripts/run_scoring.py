import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.scoring import ScoringService
from src.orchestrator import OlistPipeline



def main():
    p = OlistPipeline()
    svc = ScoringService(p.config.inference.model_dump(), p.paths.outputs.eps_dir)
    svc.run_scoring_pipeline(
        p.paths.data.processed_olist / "features_weekly.csv",
        p.paths.data.processed_olist / "prediction_data.csv"
    )

if __name__ == "__main__":
    main()
