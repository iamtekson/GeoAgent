# -*- coding: utf-8 -*-
"""
Logging module for GeoAgent plugin with UI integration.

This module provides logging functionality that displays messages in the QGIS UI
instead of saving to files.
"""

import logging
from typing import Optional, Callable
from datetime import datetime
from qgis.PyQt.QtWidgets import QTextBrowser
from qgis.PyQt.QtCore import QObject, pyqtSignal


class UILogSignal(QObject):
    """Signal emitter for logging to avoid thread issues."""
    log_signal = pyqtSignal(str)


class UILogHandler(logging.Handler):
    """
    Custom logging handler that displays log messages in a QTextBrowser widget.
    
    Features:
    - Displays log messages in UI with timestamp and level
    - Limits the number of lines displayed (default: 1000 lines)
    - Thread-safe using Qt signals
    - Colorizes log levels for better visibility
    """
    
    # Color codes for different log levels
    LOG_COLORS = {
        "DEBUG": "#808080",      # Gray
        "INFO": "#000000",       # Black
        "WARNING": "#FF8C00",    # Dark Orange
        "ERROR": "#FF0000",      # Red
        "CRITICAL": "#FF0000",   # Red
    }
    
    def __init__(
        self,
        text_browser: Optional[QTextBrowser] = None,
        max_lines: int = 1000,
        show_debug: bool = False,
    ):
        """
        Initialize UILogHandler.
        
        Args:
            text_browser: QTextBrowser widget to display logs. Can be set later with set_text_browser()
            max_lines: Maximum number of lines to keep in the text browser (default: 1000)
            show_debug: Whether to show DEBUG level messages (default: False)
        """
        super().__init__()
        self.text_browser = text_browser
        self.max_lines = max_lines
        self.show_debug = show_debug
        self.line_count = 0
        
        # Create signal emitter for thread safety
        self._signal_emitter = UILogSignal()
        self._signal_emitter.log_signal.connect(self._append_to_browser)
        
        # Set formatter
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    
    def set_text_browser(self, text_browser: QTextBrowser) -> None:
        """
        Set or update the QTextBrowser widget for displaying logs.
        
        Args:
            text_browser: QTextBrowser widget to display logs
        """
        self.text_browser = text_browser
    
    def set_show_debug(self, show_debug: bool) -> None:
        """
        Set whether to show DEBUG level messages.
        
        Args:
            show_debug: True to show DEBUG messages, False to hide them
        """
        self.show_debug = show_debug
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the UI.
        
        This method is called by the logging framework and emits the formatted
        message to the QTextBrowser through a Qt signal.
        
        Args:
            record: LogRecord to emit
        """
        # Filter out DEBUG messages if not in debug mode
        if record.levelno == logging.DEBUG and not self.show_debug:
            return
        
        try:
            # Format the message
            msg = self.format(record)
            
            # Emit signal for thread-safe UI update
            self._signal_emitter.log_signal.emit(msg)
        except Exception:
            self.handleError(record)
    
    def _append_to_browser(self, msg: str) -> None:
        """
        Append a formatted message to the text browser.
        
        This method is called from the main thread via Qt signal.
        
        Args:
            msg: Formatted log message
        """
        if self.text_browser is None:
            return
        
        try:
            # Extract log level from message for coloring
            level = self._extract_log_level(msg)
            color = self.LOG_COLORS.get(level, "#000000")
            
            # Create HTML formatted message with color
            html_msg = f'<font color="{color}">{msg}</font><br>'
            
            # Append to text browser
            self.text_browser.insertHtml(html_msg)
            
            # Update line count
            self.line_count += 1
            
            # Trim lines if exceeding max_lines
            if self.line_count > self.max_lines:
                self._trim_lines()
            
            # Scroll to bottom
            self.text_browser.verticalScrollBar().setValue(
                self.text_browser.verticalScrollBar().maximum()
            )
        except Exception:
            pass
    
    def _extract_log_level(self, msg: str) -> str:
        """
        Extract log level from formatted message.
        
        Args:
            msg: Formatted log message
            
        Returns:
            Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        for level in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
            if f"[{level}]" in msg:
                return level
        return "INFO"
    
    def _trim_lines(self) -> None:
        """
        Remove oldest lines when exceeding max_lines limit.
        
        Removes approximately 10% of max_lines to reduce trimming frequency.
        """
        if self.text_browser is None:
            return
        
        try:
            # Calculate lines to remove (remove 10% of max_lines at a time)
            lines_to_remove = max(1, self.max_lines // 10)
            
            # Get text cursor
            cursor = self.text_browser.textCursor()
            
            # Move to start
            cursor.movePosition(cursor.Start)
            
            # Select and delete lines
            for _ in range(lines_to_remove):
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # Remove the newline
            
            # Update line count
            self.line_count = self.text_browser.document().lineCount()
        except Exception:
            pass


def get_logger(
    name: str,
    text_browser: Optional[QTextBrowser] = None,
    show_debug: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger instance with UI support.
    
    This function creates or retrieves a logger with both console and UI handlers.
    
    Args:
        name: Logger name (typically __name__)
        text_browser: Optional QTextBrowser for UI logging
        show_debug: Whether to show DEBUG messages (default: False)
        level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger("GeoAgent.MyModule", text_browser, show_debug=True)
        >>> logger.info("This message will appear in both console and UI")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # UI handler
    if text_browser is not None:
        ui_handler = UILogHandler(text_browser, show_debug=show_debug)
        ui_handler.setLevel(level)
        logger.addHandler(ui_handler)
    
    return logger
