"""Centralized LLM client factory.

Ensures all LLM calls (Evidence Extraction, Forecasting, Ensemble)
respect the configured provider (OpenRouter, OpenAI, Anthropic, Google)
and use the correct API keys and base URLs.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Optional
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai

from src.observability.logger import get_logger

log = get_logger(__name__)


def get_llm_provider(model: str) -> str:
    """Determine which provider a model name belongs to."""
    model_lower = model.lower()
    if "/" in model_lower:
        # OpenRouter models use org/model format (e.g. google/gemini-pro)
        return "openrouter"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    if model_lower.startswith("gpt-"):
        return "openai"
    # Default to OpenAI for unrecognized models (legacy behavior)
    return "openai"


def create_llm_client(model: str) -> Any:
    """Create the appropriate LLM client for the given model."""
    provider = get_llm_provider(model)

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            log.warning("llm.missing_key", provider="openrouter")
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/polymarket-bot",
                "X-Title": "Polymarket Trading Bot",
            }
        )

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            log.warning("llm.missing_key", provider="anthropic")
        return anthropic.AsyncAnthropic(api_key=api_key)

    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            log.warning("llm.missing_key", provider="google")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)

    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        # Fallback to OpenRouter if OpenAI key is missing but OpenRouter key exists
        if not api_key and os.environ.get("OPENROUTER_API_KEY"):
            log.info("llm.openai_fallback_to_openrouter", model=model)
            return AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        
        if not api_key:
            log.warning("llm.missing_key", provider="openai")
        return AsyncOpenAI(api_key=api_key)
