# -*- coding: utf-8 -*-
"""Processing-specific logger helpers for GeoAgent.

This centralizes the processing logger and its UI handler wiring so that
logging concerns stay within the logger package.
"""

import logging
from typing import Optional

from ..config.settings import SHOW_DEBUG_LOGS
from .logger import get_logger, UILogHandler


# Create processing logger (no UI handler attached here; UI is wired at runtime)
_processing_logger = get_logger(
    "GeoAgent.Processing",
    text_browser=None,
    show_debug=SHOW_DEBUG_LOGS,
    level=logging.DEBUG if SHOW_DEBUG_LOGS else logging.INFO,
)
# Let messages propagate to root; root will carry the UI handler
_processing_logger.propagate = True


def get_processing_logger() -> logging.Logger:
    """Return the shared processing logger instance."""
    return _processing_logger


def set_processing_ui_log_handler(ui_log_handler: Optional[UILogHandler]) -> None:
    """Ensure processing logs flow to the shared UI handler via root.

    We avoid attaching the UI handler directly to prevent duplicate delivery; the
    handler is expected to be registered on the root logger upstream.
    """
    global _processing_logger
    if ui_log_handler is None:
        return

    # Remove any direct attachment of this UI handler to avoid duplicates
    _processing_logger.handlers = [h for h in _processing_logger.handlers if h is not ui_log_handler]
    _processing_logger.propagate = True
