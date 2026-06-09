"""
Custom exceptions for the Olist Pipeline.
"""

class OlistPipelineError(Exception):
    """Base exception for all errors in the Olist Pipeline."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ConfigurationError(OlistPipelineError):
    """Raised when there is an issue with the configuration."""
    pass

class DataPipelineError(OlistPipelineError):
    """Base exception for data-related errors."""
    pass

class DataValidationError(DataPipelineError):
    """Raised when data fails validation checks (e.g., schema mismatch)."""
    pass

class DataIngestionError(DataPipelineError):
    """Raised when data cannot be loaded from source (e.g., Kaggle, CSV)."""
    pass

class ModelError(OlistPipelineError):
    """Raised when there is an issue during model training or inference."""
    pass

class ModelTrainingError(ModelError):
    """Raised specifically during model training phase."""
    pass

class ModelInferenceError(ModelError):
    """Raised specifically during model prediction phase."""
    pass

class InfrastructureError(OlistPipelineError):
    """Raised when external services (e.g., API, DB, network) fail."""
    pass

class XAIError(OlistPipelineError):
    """Raised when explainability pipelines (SHAP, Gemini) fail."""
    pass
