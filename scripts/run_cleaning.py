from src.olist_pipeline.data.cleaning import DataService
from src.olist_pipeline.pipeline import OlistPipeline


def main():
    pipeline = OlistPipeline()
    svc = DataService(pipeline.paths.data.raw_olist, pipeline.paths.data.processed_olist)
    svc.run_cleaning_pipeline()

if __name__ == "__main__":
    main()
