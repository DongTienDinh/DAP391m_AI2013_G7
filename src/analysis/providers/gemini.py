from typing import Any

from src.analysis.providers.base import LLMProvider
from src.core.logging_setup import setup_logger

logger = setup_logger("gemini_provider")


class GeminiProvider(LLMProvider):
    """Google Gemini implementation of LLMProvider."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_narrative(self, data: dict[str, Any], system_prompt: str) -> dict[str, str]:
        """Calls Gemini API to generate XAI explanations. Falls back to placeholder on failure."""
        if not self.api_key:
            return {"brief": "API Key missing", "full": "API Key missing"}

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            user_prompt = f"Data: {data}\nGenerate JSON: {{'brief': '<under 50 words>', 'full': '<under 150 words>'}}"
            response = client.models.generate_content(
                model=self.model_name,
                contents=[system_prompt, user_prompt],
            )
            if response and response.text:
                text = response.text.strip().strip("`json\n ")
                import json
                return json.loads(text)
            return {"brief": "Empty response", "full": "Empty response from API"}
        except Exception as e:
            logger.error(f"Gemini API call failed for {data.get('state')}: {e}")
            return {"brief": "LLM Failure", "full": str(e)}
