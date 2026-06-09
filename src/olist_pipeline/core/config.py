from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.olist_pipeline.core.exceptions import ConfigurationError


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[3]


class DataPaths(BaseSettings):
    raw_olist: Path = Field(default=Path("data/raw/olist"))
    processed_olist: Path = Field(default=Path("data/processed/olist"))
    external_pop: Path = Field(default=Path("data/external/br_ibge_populacao_uf.csv"))
    external_gdp: Path = Field(default=Path("data/external/br_ibge_pib_uf.csv"))
    external_geojson: Path = Field(default=Path("data/external/br_states.geojson"))

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v: Any) -> Path:
        root = get_project_root()
        if isinstance(v, str):
            path = Path(v)
            return path if path.is_absolute() else root / path
        return v


class OutputPaths(BaseSettings):
    eps_dir: Path = Field(default=Path("outputs/eps"))
    eps_results: Path = Field(default=Path("outputs/eps/eps_results.csv"))
    w_star: Path = Field(default=Path("outputs/eps/w_star.json"))
    xai_report_json: Path = Field(default=Path("outputs/eps/eps_xai_report.json"))
    xai_report_csv: Path = Field(default=Path("outputs/eps/eps_xai_report.csv"))
    shap_profiles: Path = Field(default=Path("outputs/eps/shap/shap_state_profiles.json"))
    shap_dir: Path = Field(default=Path("outputs/eps/shap"))

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v: Any) -> Path:
        root = get_project_root()
        if isinstance(v, str):
            path = Path(v)
            return path if path.is_absolute() else root / path
        return v


class ReportPaths(BaseSettings):
    figures_dir: Path = Field(default=Path("reports/figures"))
    leaderboard: Path = Field(default=Path("reports/model_leaderboard.csv"))

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v: Any) -> Path:
        root = get_project_root()
        if isinstance(v, str):
            path = Path(v)
            return path if path.is_absolute() else root / path
        return v


class PathsConfig(BaseSettings):
    """Grouped configuration for data and file paths."""

    data: DataPaths = Field(default_factory=DataPaths)
    outputs: OutputPaths = Field(default_factory=OutputPaths)
    reports: ReportPaths = Field(default_factory=ReportPaths)


class TrainingConfig(BaseSettings):
    """Configuration for model training."""

    random_state: int = 42
    n_splits: int = 5
    log_target: bool = True
    feature_selection: dict[str, list[str]] = Field(default_factory=dict)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)


class InferenceConfig(BaseSettings):
    """Configuration for inference and scoring."""

    gamma: float = 0.20
    min_sellers: int = 5
    n_weeks: int = 4
    scoring: dict[str, Any] = Field(default_factory=dict)
    xai: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OLIST_", env_nested_delimiter="__", extra="ignore"
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
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
                paths=PathsConfig(**raw_paths),
                training=TrainingConfig(**raw_training),
                inference=InferenceConfig(**raw_inference),
            )

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to load application configuration: {e}")

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        with open(path) as f:
            return yaml.safe_load(f) or {}


# Global runtime configuration
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Returns the global application configuration singleton."""
    return _ConfigStore.get()


class _ConfigStore:
    _instance: AppConfig | None = None

    @classmethod
    def get(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = AppConfig.load()
        return cls._instance
