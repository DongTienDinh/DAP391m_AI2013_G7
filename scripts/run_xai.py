import os

from src.olist_pipeline.analysis.providers.gemini import GeminiProvider
from src.olist_pipeline.analysis.xai import XAIService
from src.olist_pipeline.pipeline import OlistPipeline


def main():
    p = OlistPipeline()
    llm = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY", ""))
    svc = XAIService(p.paths.outputs.eps_dir, llm_provider=llm)
    svc.run_xai_pipeline(p.paths.outputs.eps_results, p.paths.outputs.w_star)

if __name__ == "__main__":
    main()
