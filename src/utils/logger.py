"""Logging setup for the application.

Modules log through `logging.getLogger(__name__)`; `setup_logger()` configures
the shared handler and level once, at application start-up.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "google_genai", "faiss")


def setup_logger(level: str | int | None = None) -> logging.Logger:
    """Configure logging for the `src` package and return its logger.

    `level` defaults to LOG_LEVEL from settings. Calling this again only updates
    the level; it never adds duplicate handlers (Streamlit re-runs scripts often).
    """
    if level is None:
        from src.core.config import get_settings

        level = get_settings().log_level
    if isinstance(level, str):
        level = level.upper()

    logger = logging.getLogger("src")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger
