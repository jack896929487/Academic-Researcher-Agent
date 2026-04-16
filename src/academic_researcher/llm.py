from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from academic_researcher.net import sanitize_dead_local_proxies

# Process-level singleton: keyed by (model_name, temperature).
# A single run always uses the same config, so one instance is enough.
_model_cache: dict[tuple[str, float], BaseChatModel] = {}


def get_chat_model(*, force_new: bool = False) -> BaseChatModel:
    """
    Return a configured chat model, reusing a cached instance when possible.

    The instance is cached per (model_name, temperature) pair, so the
    expensive operations of reading .env, validating keys, and constructing
    the client object only happen once per process.

    Parameters
    ----------
    force_new : bool
        Pass True to bypass the cache and create a fresh instance
        (useful in tests or when credentials change at runtime).
    """
    load_dotenv(override=False)
    sanitize_dead_local_proxies()

    model_name  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    cache_key   = (model_name, temperature)

    if not force_new and cache_key in _model_cache:
        return _model_cache[cache_key]

    openai_api_key = os.getenv("OPENAI_API_KEY") or None
    google_api_key = os.getenv("GOOGLE_API_KEY") or openai_api_key

    if "gemini" in model_name.lower():
        if not google_api_key:
            raise ValueError(
                "Gemini model selected but no API key found. "
                "Set `GOOGLE_API_KEY` in .env (or keep OPENAI_API_KEY as fallback)."
            )
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
        )
    else:
        if not openai_api_key:
            raise ValueError(
                "OpenAI model selected but `OPENAI_API_KEY` is missing. "
                "Set it in your .env file."
            )
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
        )

    _model_cache[cache_key] = model
    return model
