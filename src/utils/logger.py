import logging

from src.core.logging_setup import setup_logger as _setup_logger


def setup_logger(name: str = "olist_pipeline") -> logging.Logger:
    """
    Configures and returns a standardized logger for the project.
    Backward compatibility wrapper for src.olist_pipeline.core.logging_setup.
    """
    return _setup_logger(name=name)
