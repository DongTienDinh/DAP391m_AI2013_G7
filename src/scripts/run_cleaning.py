import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaning import DataService
from src.orchestrator import OlistPipeline




def main():
    pipeline = OlistPipeline()
    svc = DataService(pipeline.paths.data.raw_olist, pipeline.paths.data.processed_olist)
    svc.run_cleaning_pipeline()

if __name__ == "__main__":
    main()
