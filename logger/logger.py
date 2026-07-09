# -*- coding: utf-8 -*-
"""
Logging module for GeoAgent plugin.

Provides the unified "geo_agent" logger: messages are written to a rotating
log file under the QGIS profile directory (<profile>/GeoAgent/geo_agent.log),
echoed to the console, and mirrored to the plugin's Logs tab via UILogHandler.
"""

import os
from qgis.core import QgsApplication
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from qgis.PyQt.QtWidgets import QTextBrowser
from qgis.PyQt.QtCore import QObject, pyqtSignal
from qgis.PyQt.QtGui import QTextCursor


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
    ):
        """
        Initialize UILogHandler.

        Args:
            text_browser: QTextBrowser widget to display logs. Can be set later with set_text_browser()
            max_lines: Maximum number of lines to keep in the text browser (default: 1000)
        """
        super().__init__()
        self.text_browser = text_browser
        self.max_lines = max_lines
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
        
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the UI.
        
        This method is called by the logging framework and emits the formatted
        message to the QTextBrowser through a Qt signal.
        
        Args:
            record: LogRecord to emit
        """
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
            cursor.movePosition(QTextCursor.MoveOperation.Start)

            # Select and delete lines
            for _ in range(lines_to_remove):
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # Remove the newline
            
            # Update line count
            self.line_count = self.text_browser.document().lineCount()
        except Exception:
            pass

def _get_log_file_path() -> str:
    base = QgsApplication.qgisSettingsDirPath()
    log_dir = os.path.join(base, "GeoAgent")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "geo_agent.log")

def configure_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Configure the unified "geo_agent" logger.

    Attaches a rotating file handler (<profile>/GeoAgent/geo_agent.log) and a
    console handler. Safe to call repeatedly (e.g. on plugin reload): existing
    handlers are updated in place instead of duplicated.
    """
    logger = logging.getLogger("geo_agent")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = None
    console_handler = None
    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            file_handler = h
        elif isinstance(h, logging.FileHandler):
            # Stale non-rotating handler left over from an older plugin version
            logger.removeHandler(h)
            h.close()
        elif isinstance(h, logging.StreamHandler):
            console_handler = h

    if file_handler is None:
        file_handler = RotatingFileHandler(
            _get_log_file_path(),
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    file_handler.setLevel(level)

    if console_handler is None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    console_handler.setLevel(level)

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the unified "geo_agent" logger, or a "geo_agent.<name>" child."""
    if name:
        return logging.getLogger(f"geo_agent.{name}")
    return logging.getLogger("geo_agent")

def attach_ui_handler(text_browser: QTextBrowser) -> None:
    logger = logging.getLogger("geo_agent")

    for h in logger.handlers:
        if isinstance(h, UILogHandler):
            h.set_text_browser(text_browser)
            return

    ui_handler = UILogHandler(text_browser)
    ui_handler.setLevel(logger.level)
    logger.addHandler(ui_handler)
