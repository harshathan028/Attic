"""
LLM Client - Abstract provider layer for language model interactions.

This module provides an abstraction over LLM providers, supporting:
- Google Gemini API (cloud)
- Ollama (local)
"""

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0-2).

        Returns:
            The generated text response.
        """
        pass


class GeminiClient(LLMClient):
    """
    Google Gemini LLM client.
    
    Uses the google-genai library to interact with Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
            model: Model identifier to use (e.g., gemini-2.0-flash, gemini-1.5-pro).
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
        """
        from google import genai
        from google.genai import types
        
        self.genai = genai
        self.types = types
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure API key
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized Gemini client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response using the Gemini API.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0-2).

        Returns:
            The generated text response.

        Raises:
            Exception: If all retry attempts fail.
        """
        # Build contents
        contents = prompt
        
        # Configure generation settings
        config = self.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_prompt,
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                result = response.text
                logger.debug(f"LLM response received: {len(result)} characters")
                return result

            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                
                # Parse retry delay from error if available
                retry_match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                if retry_match and attempt < self.max_retries - 1:
                    sleep_time = float(retry_match.group(1)) + 1
                    logger.info(f"Rate limited. Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                elif attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

        logger.error(f"All LLM retry attempts failed: {last_error}")
        raise last_error


class OllamaClient(LLMClient):
    """
    Ollama LLM client for local models.
    
    Connects to a local Ollama server to run LLMs without API limits.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 120,
    ):
        """
        Initialize the Ollama client.

        Args:
            model: Model name (e.g., llama3.2, mistral, qwen3:4b).
            base_url: Ollama server URL.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Check if Ollama is available
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                logger.info(f"Ollama available with models: {models}")
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
        except requests.ConnectionError:
            logger.warning(f"Cannot connect to Ollama at {base_url}. Make sure Ollama is running.")
        
        logger.info(f"Initialized Ollama client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response using Ollama.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            The generated text response.

        Raises:
            Exception: If all retry attempts fail.
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Ollama request attempt {attempt + 1}/{self.max_retries}")
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                result = response.json().get("response", "")
                logger.debug(f"Ollama response received: {len(result)} characters")
                return result

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"Ollama connection failed: {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Ollama request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

        logger.error(f"All Ollama retry attempts failed: {last_error}")
        raise last_error


# Alias for backward compatibility
OpenAIClient = GeminiClient


class MockLLMClient(LLMClient):
    """Mock LLM client for testing purposes."""

    def __init__(self, responses: Optional[dict] = None):
        """
        Initialize mock client with predefined responses.

        Args:
            responses: Dictionary mapping prompt keywords to responses.
        """
        self.responses = responses or {}
        self.call_history = []
        logger.info("Initialized MockLLMClient")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """Return a mock response based on the prompt."""
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        # Check for keyword matches
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return response

        # Default mock response
        return f"[Mock LLM Response for prompt: {prompt[:100]}...]"


def create_llm_client(
    provider: str = "gemini",
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: Provider name ('gemini', 'ollama', 'mock').
        model: Model name (uses default if not specified).
        **kwargs: Additional arguments for the client.
        
    Returns:
        An LLMClient instance.
    """
    provider = provider.lower()
    
    if provider == "gemini":
        return GeminiClient(model=model or "gemini-2.0-flash", **kwargs)
    elif provider == "ollama":
        return OllamaClient(model=model or "llama3.2", **kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'ollama', or 'mock'.")
