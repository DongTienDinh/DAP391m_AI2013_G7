from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate_narrative(
        self, 
        data: Dict[str, Any], 
        system_prompt: str
    ) -> Dict[str, str]:
        """Generates brief and full narratives from input data."""
        pass
