import yaml
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[3]

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the configs/ directory.
    """
    root = get_project_root()
    config_path = root / "configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Config:
    """Centralized configuration accessor."""
    paths = load_config("paths")
    training = load_config("training")
    inference = load_config("inference")
    
    @classmethod
    def get_path(cls, *keys: str) -> Path:
        """Helper to get a path and resolve it against project root."""
        val = cls.paths
        for k in keys:
            val = val[k]
        return get_project_root() / val
