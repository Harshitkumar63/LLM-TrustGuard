"""
logger.py — Centralized logging for the AI LLM Validator system.

Provides a configured ``logging.Logger`` instance that writes to both
the console (INFO+) and a rotating log file.  Every module should import
``get_logger`` and create a module-level logger:

    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from utils.config import LOG_FILE, LOG_LEVEL

# ──────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str, optional
        Logger name — typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Logger with console and file handlers attached.
    """
    logger = logging.getLogger(name or "ai_llm_validator")

    # Avoid duplicate handlers if called multiple times for the same name
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler (rotating, max 5 MB × 3 backups) ---
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file at %s", LOG_FILE)

    return logger
