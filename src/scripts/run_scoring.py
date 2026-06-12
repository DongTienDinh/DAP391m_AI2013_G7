import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.scoring import ScoringService
from src.orchestrator import OlistPipeline



def main():
    print()
    print("  >>> EPS SCORING & RANKING")
    print("  >>> Calculating Expansion Priority Scores (SLSQP optimization)...")
    p = OlistPipeline()
    svc = ScoringService(p.config.inference.model_dump(), p.paths.outputs.eps_dir)
    svc.run_scoring_pipeline(
        p.paths.data.processed_olist / "features_weekly.csv",
        p.paths.data.processed_olist / "prediction_data.csv"
    )
    print("  ✅ EPS Scoring completed.")

if __name__ == "__main__":
    main()
