from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    api_key: str = ""

    @abstractmethod
    def generate_narrative(self, data: dict[str, Any], system_prompt: str) -> dict[str, str]:
        """Generates brief and full narratives from input data."""
        pass
