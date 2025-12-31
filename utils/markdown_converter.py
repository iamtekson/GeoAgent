# -*- coding: utf-8 -*-
"""
Markdown to HTML converter utilities for GeoAgent.
"""

import logging

_logger = logging.getLogger(__name__)

try:
    # try to use the markdown library if available
    import markdown
    HAS_MARKDOWN = True
    _logger.info("markdown library loaded successfully")
except ImportError:
    # fallback to manual conversion
    HAS_MARKDOWN = False
    import re
    import html
    _logger.warning("markdown library not available, using fallback regex conversion")


def markdown_to_html(text: str) -> str:
    """
    Convert Markdown formatting to HTML.
    
    Uses the markdown library if available for full Markdown support,
    otherwise falls back to basic regex-based conversion for bold/italic.
    
    Args:
        text: Markdown text
        
    Returns:
        HTML texts
    """
    if HAS_MARKDOWN:
        # use full markdown library for comprehensive support
        return markdown.markdown(
            text,
            extensions=['nl2br', 'sane_lists'],  # Better line break and list handling
        )
    else:
        # fallback: basic conversion using regex
        text = html.escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)  # **bold**
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)      # *italic*
        text = text.replace('\n', '<br>')
        return text


__all__ = ["markdown_to_html"]
