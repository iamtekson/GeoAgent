# -*- coding: utf-8 -*-
"""Sphinx configuration for the GeoAgent documentation (Read the Docs)."""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

project = "GeoAgent"
author = "Tek Kshetri, Rabin Ojha"
copyright = "2025-2026, Tek Kshetri and Rabin Ojha"

# The plugin's metadata.txt is the single source of the version number
with open(os.path.join(_HERE, "..", "metadata.txt"), encoding="utf-8") as _f:
    release = re.search(r"^version=(.+)$", _f.read(), re.MULTILINE).group(1).strip()
version = release

extensions = [
    "myst_parser",  # pages are written in Markdown
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "scripts", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"GeoAgent {release}"
html_logo = "../icons/icon.png"
html_favicon = "../icons/icon.png"
html_theme_options = {
    "source_repository": "https://github.com/iamtekson/GeoAgent/",
    "source_branch": "main",
    "source_directory": "docs/",
}
