import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.providers.gemini import GeminiProvider
from src.analysis.xai import XAIService
from src.orchestrator import OlistPipeline


def main():
    print()
    print("  >>> XAI NARRATIVE GENERATION")
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    print(f"  >>> {'Gemini API' if has_key else 'Rule-based'} explanations for 27 states...")
    p = OlistPipeline()
    llm = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY", ""))
    svc = XAIService(p.paths.outputs.eps_dir, llm_provider=llm)
    svc.run_xai_pipeline(p.paths.outputs.eps_results, p.paths.outputs.w_star)
    print("  ✅ XAI narratives completed.")

if __name__ == "__main__":
    main()
