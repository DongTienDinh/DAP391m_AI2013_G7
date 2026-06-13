import logging
import sys
from pathlib import Path


def short_path(p: Path) -> str:
    """Return a short relative path for display."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def setup_logger(
    name: str = "olist_pipeline", level: int = logging.INFO, log_format: str | None = None
) -> logging.Logger:
    """Sets up a clean, human-readable logger for the CLI pipeline."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if not log_format:
        log_format = "  %(message)s"

    handler.setFormatter(logging.Formatter(fmt=log_format))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
