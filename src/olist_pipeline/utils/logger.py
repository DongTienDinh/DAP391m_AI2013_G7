import logging
import sys

def setup_logger(name: str = "olist_pipeline") -> logging.Logger:
    """
    Configures and returns a standardized logger for the project.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Standard console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Consistent format: [TIMESTAMP] LEVEL - MODULE - MESSAGE
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
