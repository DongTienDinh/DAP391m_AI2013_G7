"""
Backward compatibility wrapper for the new configuration system.
"""

from pathlib import Path
from typing import Any


from src.olist_pipeline.core.config import get_config, get_project_root


class _ConfigMeta(type):
    """Metaclass to support class-level property access for backward compatibility."""

    @property
    def paths(cls) -> dict[str, Any]:
        return get_config().paths.model_dump()

    @property
    def training(cls) -> dict[str, Any]:
        return get_config().training.model_dump()

    @property
    def inference(cls) -> dict[str, Any]:
        return get_config().inference.model_dump()


class Config(metaclass=_ConfigMeta):
    """
    Centralized configuration accessor.
    Refactored to use the validated core.config system.
    """

    @classmethod
    def get_path(cls, *keys: str) -> Path:
        """Helper to get a path and resolve it against project root."""
        config = get_config()
        # Navigate the nested structure
        data = config.paths
        for k in keys:
            if hasattr(data, k):
                data = getattr(data, k)
            elif isinstance(data, dict) and k in data:
                data = data[k]
            else:
                # Fallback to dict if it's a sub-model or dict
                data = data.model_dump()[k]

        if isinstance(data, Path):
            return data
        # If it's a sub-model or dict, and we still have a key, we might have an issue.
        # But for the base case, if it's a string, return Path.
        return get_project_root() / str(data)