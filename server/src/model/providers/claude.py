"""Anthropic Claude provider.

Import `ClaudeProvider` directly only when you need to instantiate a specific
model by name. For all application code, prefer:

    from src.model.providers.registry import primary, cheap
"""

import anthropic

from src.config import settings
from src.model.providers.base import LLMProvider, LLMResponse


class ClaudeProvider(LLMProvider):
    """Wraps the Anthropic Messages API."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = self._get_client().messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
