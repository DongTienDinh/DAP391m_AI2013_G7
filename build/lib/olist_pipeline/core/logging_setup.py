import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "olist_pipeline", 
    level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Sets up a standardized, structured logger for the project.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO).
        log_format: Optional custom format string.
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    
    # Console Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Default Production-ready format
    if not log_format:
        log_format = '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
    
    formatter = logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger
