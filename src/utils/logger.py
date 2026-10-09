"""Logging utility for structured pipeline and service logs."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "retail_intelligence", log_file: Optional[str] = "outputs/pipeline.log", level: str = "INFO") -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(numeric_level)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        f_handler.setLevel(numeric_level)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.propagate = False
    return logger
