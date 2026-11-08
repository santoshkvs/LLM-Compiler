"""LLM provider adapters. The core library has no hard dependency on any SDK."""

from __future__ import annotations

import time
from typing import Callable, Optional

from .errors import ProviderError


class Provider:
    """Minimal interface: send a prompt, get text back."""

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        raise NotImplementedError


class CallableProvider(Provider):
    """Wrap ``fn(prompt, temperature, seed) -> str``. Handy for tests and custom backends."""

    def __init__(self, fn: Callable[[str, float, Optional[int]], str]):
        self._fn = fn

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        return self._fn(prompt, temperature, seed)


class AnthropicProvider(Provider):
    """Claude via the ``anthropic`` SDK (``pip install decisionforge[anthropic]``).

    Reads ``ANTHROPIC_API_KEY`` from the environment unless ``api_key`` is given.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, max_tokens: int = 64, system: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("pip install 'decisionforge[anthropic]'") from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system = system

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.system:
            kwargs["system"] = self.system
        msg = self._client.messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in msg.content)


class OpenAIProvider(Provider):
    """OpenAI or any OpenAI-compatible server (Ollama, vLLM, ...) via ``openai``.

    ``token_param`` is the name of the output-length parameter; some newer OpenAI
    models want ``max_completion_tokens`` instead of ``max_tokens``.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 64,
        token_param: str = "max_tokens",
        system: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ProviderError("pip install 'decisionforge[openai]'") from exc
        kw = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        self._client = openai.OpenAI(**kw)
        self.model = model
        self.max_tokens = max_tokens
        self.token_param = token_param
        self.system = system

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        kwargs[self.token_param] = self.max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class RetryingProvider(Provider):
    """Retry transient failures with exponential backoff."""

    def __init__(self, inner: Provider, retries: int = 3, backoff: float = 1.0):
        self.inner = inner
        self.retries = retries
        self.backoff = backoff

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return self.inner.complete(prompt, temperature=temperature, seed=seed)
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        raise ProviderError(f"provider failed after {self.retries + 1} attempts: {last}") from last
