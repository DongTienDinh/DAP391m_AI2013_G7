import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.olist_pipeline.core.exceptions import ConfigurationError

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[3]

class PathConfig(BaseSettings):
    """Configuration for data and file paths."""
    raw_olist: Path = Field(default=Path("data/raw/olist"))
    processed_olist: Path = Field(default=Path("data/processed/olist"))
    external_pop: Path = Field(default=Path("data/external/br_ibge_populacao_uf.csv"))
    external_gdp: Path = Field(default=Path("data/external/br_ibge_pib_uf.csv"))
    external_geojson: Path = Field(default=Path("data/external/br_states.geojson"))
    
    eps_dir: Path = Field(default=Path("outputs/eps"))
    eps_results: Path = Field(default=Path("outputs/eps/eps_results.csv"))
    w_star: Path = Field(default=Path("outputs/eps/w_star.json"))
    xai_report_json: Path = Field(default=Path("outputs/eps/eps_xai_report.json"))
    xai_report_csv: Path = Field(default=Path("outputs/eps/eps_xai_report.csv"))
    shap_profiles: Path = Field(default=Path("outputs/eps/shap/shap_state_profiles.json"))
    shap_dir: Path = Field(default=Path("outputs/eps/shap"))
    
    figures_dir: Path = Field(default=Path("reports/figures"))
    leaderboard: Path = Field(default=Path("reports/model_leaderboard.csv"))

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v: Any) -> Path:
        """Ensures all paths are absolute and relative to project root."""
        root = get_project_root()
        if isinstance(v, str):
            path = Path(v)
            return path if path.is_absolute() else root / path
        return v

class TrainingConfig(BaseSettings):
    """Configuration for model training."""
    random_state: int = 42
    n_splits: int = 5
    log_target: bool = True
    feature_selection: Dict[str, List[str]] = Field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class InferenceConfig(BaseSettings):
    """Configuration for inference and scoring."""
    gamma: float = 0.20
    min_sellers: int = 5
    n_weeks: int = 4
    scoring: Dict[str, Any] = Field(default_factory=dict)
    xai: Dict[str, Any] = Field(default_factory=dict)

class AppConfig(BaseSettings):
    """Main application configuration."""
    model_config = SettingsConfigDict(
        env_prefix="OLIST_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def load(cls) -> "AppConfig":
        """
        Loads configuration from YAML files and environment variables.
        Environment variables take precedence.
        """
        root = get_project_root()
        configs_dir = root / "configs"
        
        try:
            # Load YAML files manually to inject into Pydantic
            raw_paths = cls._load_yaml(configs_dir / "paths.yaml")
            raw_training = cls._load_yaml(configs_dir / "training.yaml")
            raw_inference = cls._load_yaml(configs_dir / "inference.yaml")
            
            return cls(
                paths=raw_paths,
                training=raw_training,
                inference=raw_inference
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load application configuration: {e}")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

# Global runtime configuration
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Returns the global application configuration singleton."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config
