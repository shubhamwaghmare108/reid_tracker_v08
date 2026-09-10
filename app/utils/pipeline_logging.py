"""Logging setup for the video processing pipeline."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

LOGGER_NAME = 'person_reid.pipeline'


class _UtcFormatter(logging.Formatter):
    converter = staticmethod(lambda timestamp: datetime.fromtimestamp(timestamp, timezone.utc).timetuple())


def get_pipeline_logger() -> logging.Logger:
    """Return the shared pipeline logger without changing application-wide logging."""
    return logging.getLogger(LOGGER_NAME)


def configure_pipeline_logging(log_dir: Path) -> Path:
    """Configure rotating file and console handlers, returning the log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'pipeline.log'
    logger = get_pipeline_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = _UtcFormatter('%(asctime)sZ | %(levelname)s | %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return log_path
