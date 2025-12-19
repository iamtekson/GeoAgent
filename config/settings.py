# -*- coding: utf-8 -*-
"""
Configuration settings for GeoAgent plugin.
"""
import os

# Plugin root directory
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API Keys
API_KEY_FILE = os.path.join(PLUGIN_DIR, "api_key.txt")

# Model configurations
SUPPORTED_MODELS = {
    "Ollama": {
        "type": "ollama",
        "default_url": "http://localhost:11434",
        "default_model": "llama3.2:3b",
        "requires_api_key": False,
    },
    "ChatGPT": {
        "type": "openai",
        "default_model": "gpt-5",
        "requires_api_key": True,
    },
    "Gemini": {
        "type": "google",
        "default_model": "gemini-pro",
        "requires_api_key": True,
    },
}

# Default model
DEFAULT_MODEL = "Ollama"

# Temperature settings
DEFAULT_TEMPERATURE = 0.8
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0

# Token settings
DEFAULT_MAX_TOKENS = 3000
MAX_ALLOWED_TOKENS = 10000

# Chat history
MAX_HISTORY_LENGTH = 10
