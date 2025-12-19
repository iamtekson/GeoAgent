# -*- coding: utf-8 -*-
"""
LLM client factory for supporting multiple LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a query to the LLM.

        :param messages: List of message dicts with 'role' and 'content' keys
        :param temperature: Temperature for generation (0-2)
        :param max_tokens: Maximum tokens to generate
        :return: Generated response text
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaClient(LLMClient):
    """Ollama LLM client."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize Ollama client.

        :param base_url: Base URL for Ollama service
        :param model: Model name to use
        """
        self.base_url = base_url
        self.model = model
        try:
            import requests

            self.requests = requests
        except ImportError:
            raise ImportError("requests package required for Ollama client")
        self._server_available: Optional[bool] = None
        self._model_missing: bool = False

    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Query Ollama."""
        try:
            # Convert messages to prompt format
            prompt = self._format_messages(messages)

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }

            if max_tokens:
                payload["num_predict"] = max_tokens

            response = self.requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            raise RuntimeError(f"Ollama query failed: {str(e)}")

    def validate_connection(self) -> bool:
        """Check if Ollama is running and the requested model is available."""
        self._server_available = None
        self._model_missing = False
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self._server_available = False
                return False

            self._server_available = True
            data = response.json()
            models = []
            # Ollama returns {'models': [{ 'name': 'llama2', ... }, ...]}
            if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                for m in data["models"]:
                    # prefer 'name', fallback to 'model'
                    name = m.get("name") or m.get("model")
                    if name:
                        models.append(name)
            # If no models parsed, still consider server available
            if self.model not in models:
                self._model_missing = True
                return False
            return True
        except Exception:
            self._server_available = False
            return False

    def model_exists(self) -> bool:
        """Return True if the requested model is installed in Ollama."""
        # Leverage last validate call if present; else perform check
        if self._server_available is None:
            _ = self.validate_connection()
        return not self._model_missing

    def explain_validate_failure(self) -> str:
        """Human-friendly explanation for validation failure with install hint."""
        if self._server_available is False:
            return (
                f"Cannot reach Ollama at {self.base_url}. Ensure Ollama is running and accessible."
            )
        if self._model_missing:
            return (
                f"Ollama model '{self.model}' not found. Install it with: ollama pull {self.model}"
            )
        return "Unknown error validating Ollama connection."

    def pull_model(self) -> bool:
        """
        Attempt to pull the missing model from Ollama.
        
        :return: True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": self.model, "stream": False}
            response = self.requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                self._model_missing = False
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        return prompt


class OpenAIClient(LLMClient):
    """OpenAI ChatGPT client."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client.

        :param api_key: OpenAI API key
        :param model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        try:
            import openai

            self.openai = openai
            self.openai.api_key = api_key
        except ImportError:
            raise ImportError("openai package required for OpenAI client")

    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Query OpenAI."""
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 1000,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI query failed: {str(e)}")

    def validate_connection(self) -> bool:
        """Check if API key is valid."""
        try:
            self.openai.Model.list()
            return True
        except Exception:
            return False


class GeminiClient(LLMClient):
    """Google Gemini client."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """
        Initialize Gemini client.

        :param api_key: Google API key
        :param model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        try:
            import google.generativeai as genai

            self.genai = genai
            genai.configure(api_key=api_key)
        except ImportError:
            raise ImportError("google-generativeai package required for Gemini client")

    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Query Gemini."""
        try:
            model = self.genai.GenerativeModel(self.model)

            # Convert messages to format Gemini expects
            prompt = self._format_messages(messages)

            response = model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens or 1000,
                ),
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini query failed: {str(e)}")

    def validate_connection(self) -> bool:
        """Check if API key is valid."""
        try:
            self.genai.list_models()
            return True
        except Exception:
            return False

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string."""
        prompt = ""
        for msg in messages:
            content = msg.get("content", "")
            prompt += content + "\n"
        return prompt


def create_client(provider: str, api_key: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Factory function to create LLM clients.

    :param provider: Provider name ('ollama', 'openai', 'gemini')
    :param api_key: API key for the provider
    :param kwargs: Additional arguments for the client
    :return: LLMClient instance
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        return OllamaClient(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama3.2:3b"),
        )
    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI")
        return OpenAIClient(api_key, model=kwargs.get("model", "gpt-3.5-turbo"))
    elif provider == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini")
        return GeminiClient(api_key, model=kwargs.get("model", "gemini-pro"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
