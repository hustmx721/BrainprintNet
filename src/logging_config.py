"""Central logging configuration for BrainprintNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

def configure_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure package-wide logging handlers.

    Parameters
    ----------
    log_file:
        Optional file path used to mirror console logs to disk.

    Returns
    -------
    logging.Logger
        The configured package logger.
    """

    logger = logging.getLogger("brainprintnet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
