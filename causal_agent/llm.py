"""
causal_agent/llm.py

LLM-agnostic adapter layer.

Every concrete backend implements BaseLLM.complete(prompt, system) -> str.
Swapping models is a one-line change in configuration; no other module
is aware of which backend is in use.

Provided adapters
-----------------
MockLLM       – deterministic/canned responses; no API key needed (tests, demos).
OpenAILLM     – OpenAI chat completions (gpt-4o, etc.).
AnthropicLLM  – Anthropic messages API (claude-3-5-sonnet, etc.).
GeminiLLM     – Google Gemini generative AI (gemini-2.0-flash, etc.).
DeepSeekLLM   – DeepSeek API via OpenAI-compatible interface (deepseek-chat, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import cycle
from typing import Iterator


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    Minimal interface all LLM backends must satisfy.

    Parameters
    ----------
    prompt  : the user-turn content (already assembled by the caller).
    system  : optional system-prompt override; backends may ignore if unsupported.
    kwargs  : pass-through for backend-specific options (temperature, max_tokens…).
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "", **kwargs) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Mock — no network, no key
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """
    Returns canned responses in round-robin order.

    Useful for unit tests and offline demos.  If no responses are supplied
    the backend echoes a fixed placeholder string.
    """

    _DEFAULTS = [
        '{"intent": "gather information", "action_type": "speak", '
        '"parameters": {"message": "I have no strong read yet — who do you all suspect?"}, '
        '"reasoning": "Early game, probing for reactions."}',

        '{"intent": "cast vote", "action_type": "vote", '
        '"parameters": {"target": "Alice"}, '
        '"reasoning": "Alice has been evasive — voting to eliminate."}',

        '{"intent": "eliminate threat", "action_type": "kill", '
        '"parameters": {"target": "Bob"}, '
        '"reasoning": "Bob is most likely to expose us."}',

        '{"intent": "deflect suspicion", "action_type": "speak", '
        '"parameters": {"message": "I am confident I am a villager — let\'s focus on the quiet players."}, '
        '"reasoning": "Need to shift attention away from myself."}',
    ]

    def __init__(self, responses: list[str] | None = None) -> None:
        self._cycle: Iterator[str] = (
            cycle(responses) if responses else cycle(self._DEFAULTS)
        )

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return next(self._cycle)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """
    Thin wrapper around the openai SDK chat-completions endpoint.

    Parameters
    ----------
    model   : any OpenAI chat model slug, e.g. "gpt-4o".
    api_key : if None, the SDK reads OPENAI_API_KEY from the environment.
    kwargs  : forwarded to every client.chat.completions.create() call.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        **default_kwargs,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._defaults = default_kwargs

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {**self._defaults, **kwargs}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"OpenAILLM(model={self._model!r})"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicLLM(BaseLLM):
    """
    Thin wrapper around the anthropic SDK messages endpoint.

    Parameters
    ----------
    model      : any Anthropic model slug, e.g. "claude-3-5-sonnet-20241022".
    api_key    : if None, the SDK reads ANTHROPIC_API_KEY from the environment.
    max_tokens : default token budget per call (required by Anthropic API).
    kwargs     : forwarded to every messages.create() call.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        max_tokens: int = 1024,
        **default_kwargs,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._defaults = default_kwargs

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        params: dict = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            **self._defaults,
            **kwargs,
        }
        if system:
            params["system"] = system

        resp = self._client.messages.create(**params)
        return resp.content[0].text if resp.content else ""

    def __repr__(self) -> str:
        return f"AnthropicLLM(model={self._model!r})"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    """
    Thin wrapper around the google-genai SDK (google-generativeai >= 0.7).

    Parameters
    ----------
    model   : any Gemini model slug, e.g. "gemini-2.0-flash" or "gemini-1.5-pro".
    api_key : if None, the SDK reads GOOGLE_API_KEY from the environment.
    kwargs  : forwarded to every generate_content() call (temperature, etc.).

    System prompt handling
    ----------------------
    Gemini supports system instructions via the `system_instruction` parameter
    on the GenerativeModel constructor.  When a non-empty system string is
    passed to complete(), a fresh model instance is created with that
    instruction so per-call overrides work without storing state.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        **default_kwargs,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("pip install google-generativeai") from exc

        self._genai = genai
        if api_key:
            genai.configure(api_key=api_key)
        self._model_name = model
        self._defaults = default_kwargs
        # Default model instance (no system instruction).
        self._model = genai.GenerativeModel(model_name=model)

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        if system:
            model = self._genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
            )
        else:
            model = self._model

        params = {**self._defaults, **kwargs}
        config = self._genai.types.GenerationConfig(**params) if params else None

        resp = model.generate_content(
            prompt,
            generation_config=config,
        )
        return resp.text if resp.text else ""

    def __repr__(self) -> str:
        return f"GeminiLLM(model={self._model_name!r})"


# ---------------------------------------------------------------------------
# DeepSeek — OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

class DeepSeekLLM(BaseLLM):
    """
    Thin wrapper around the DeepSeek API using the OpenAI-compatible interface.

    DeepSeek exposes an OpenAI-compatible REST endpoint, so we reuse the
    openai SDK pointed at https://api.deepseek.com/v1.

    Parameters
    ----------
    model   : DeepSeek model slug, e.g. "deepseek-chat" or "deepseek-reasoner".
    api_key : if None, reads DEEPSEEK_API_KEY from the environment.
    kwargs  : forwarded to every chat.completions.create() call.
    """

    _BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        **default_kwargs,
    ) -> None:
        import os
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Pass api_key= or set DEEPSEEK_API_KEY in your environment."
            )

        self._client = OpenAI(api_key=resolved_key, base_url=self._BASE_URL)
        self._model = model
        self._defaults = default_kwargs

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            **self._defaults,
            **kwargs,
        }
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"DeepSeekLLM(model={self._model!r})"
