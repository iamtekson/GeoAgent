# -*- coding: utf-8 -*-
"""Processing-specific logger helpers for GeoAgent.

Provides a shared processing logger that inherits configuration
from the main GeoAgent logger.
"""

import logging
from .logger import get_logger


# Processing logger inherits handlers and level from core logger
_processing_logger = get_logger("processing")


def get_processing_logger() -> logging.Logger:
    """Return the shared processing logger instance."""
    return _processing_logger
