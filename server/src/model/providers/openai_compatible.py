"""OpenAI-compatible LLM provider.

Works with any API that implements the OpenAI chat completions format:
  - OpenAI (GPT-4o, o1, ...)
  - DeepSeek (https://api.deepseek.com/v1)
  - OpenRouter (https://openrouter.ai/api/v1)  — access to 200+ models
  - Gemini via OpenAI endpoint (https://generativelanguage.googleapis.com/v1beta/openai)
  - Any self-hosted model via vLLM / Ollama / LM Studio

Configuration in .env:
    OPENAI_API_KEY=<your key>
    OPENAI_BASE_URL=https://api.deepseek.com/v1   # or openrouter, gemini, etc.
    PRIMARY_PROVIDER=openai_compatible
    PRIMARY_MODEL=deepseek-chat
    CHEAP_PROVIDER=openai_compatible
    CHEAP_MODEL=deepseek-chat
"""

import httpx

from src.model.providers.base import LLMProvider, LLMResponse

_DEFAULT_TIMEOUT = 120.0  # seconds


class OpenAICompatibleProvider(LLMProvider):
    """Synchronous provider for any OpenAI chat-completions-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self._model = model
        self._api_key = api_key
        # Normalise: strip trailing slash so we can always append /chat/completions
        self._base_url = base_url.rstrip("/")
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_DEFAULT_TIMEOUT,
            )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat completion request to the configured OpenAI-compatible endpoint.

        The `system` string is prepended as a {"role": "system"} message.
        """
        full_messages = [{"role": "system", "content": system}, *messages]

        payload = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._get_client().post(
            f"{self._base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(
                f"OpenAI-compatible API returned no choices. "
                f"Full response: {data}"
            )

        content = choices[0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
