# -*- coding: utf-8 -*-
"""
LLM client factory and utilities for creating chat models.
"""
import importlib
import requests
from typing import Optional


def create_llm(provider: str, api_key: Optional[str] = None, **kwargs):
    """
    Create a LangChain chat model based on the provider.

    Args:
        provider: One of 'openai', 'google', 'ollama'
        api_key: API key for providers that require it (openai, google)
        **kwargs: Additional arguments like model, temperature, max_tokens, base_url

    Returns:
        Chat model instance
    """
    if provider == "openai":
        # Lazy import to avoid dependency issues
        openai_mod = importlib.import_module("langchain_openai")
        ChatOpenAI = openai_mod.ChatOpenAI

        return ChatOpenAI(
            api_key=api_key,
            model=kwargs.get("model", "gpt-3.5-turbo"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
        )

    elif provider == "google":
        # Lazy import
        google_mod = importlib.import_module("langchain_google_genai")
        ChatGoogleGenerativeAI = google_mod.ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=kwargs.get("model", "gemini-pro"),
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens"),
        )

    elif provider == "ollama":
        # Use langchain-ollama library
        ollama_mod = importlib.import_module("langchain_ollama")
        ChatOllama = ollama_mod.ChatOllama

        ollama_kwargs = {
            "model": kwargs.get("model", "llama3.2:3b"),
            "temperature": kwargs.get("temperature", 0.7),
        }

        # Optional base_url for custom Ollama server
        if kwargs.get("base_url"):
            ollama_kwargs["base_url"] = kwargs["base_url"]

        # Optional max_tokens
        if kwargs.get("max_tokens"):
            ollama_kwargs["num_predict"] = kwargs["max_tokens"]

        return ChatOllama(**ollama_kwargs)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def ollama_model_exists(base_url: str, model_name: str) -> bool:
    """
    Check if an Ollama model exists locally.

    Args:
        base_url: Ollama server URL (e.g., http://localhost:11434)
        model_name: Model name to check (e.g., llama3.2:3b)

    Returns:
        True if model exists, False otherwise
    """
    try:
        url = f"{base_url}/api/tags"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            # Check if model exists in the list
            for model in models:
                if model.get("name") == model_name:
                    return True

        return False

    except Exception:
        return False


def ollama_pull_model(base_url: str, model_name: str) -> bool:
    """
    Pull an Ollama model from the registry.

    Args:
        base_url: Ollama server URL (e.g., http://localhost:11434)
        model_name: Model name to pull (e.g., llama3.2:3b)

    Returns:
        True if pull succeeded, False otherwise
    """
    try:
        url = f"{base_url}/api/pull"
        payload = {"name": model_name, "stream": False}

        # Use a longer timeout for pulling models (can take several minutes)
        response = requests.post(url, json=payload, timeout=600)

        if response.status_code == 200:
            return True

        return False

    except Exception:
        return False
