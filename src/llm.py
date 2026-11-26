"""Provider-agnostic LLM and embedding adapter for Gemini, OpenAI, and Ollama."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import httpx
import openai
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

ProviderName = Literal["gemini", "openai", "ollama"]
T = TypeVar("T", bound=BaseModel)

_TRANSIENT_HTTP_CODES = {429, 502, 503, 504}
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class LLMError(RuntimeError):
    """Base exception for provider adapter failures."""


class MissingAPIKeyError(LLMError):
    """Raised when a required provider API key is missing."""


class UnsupportedProviderError(LLMError):
    """Raised when an unknown LLM provider name is requested."""


class StructuredOutputError(LLMError):
    """Raised when structured JSON output cannot be parsed or validated."""


class LLMConfig(BaseModel):
    """Resolved runtime configuration for an LLM and embedding provider."""

    provider: ProviderName = "gemini"
    api_key: str | None = None
    base_url: str | None = None
    chat_model: str
    embed_model: str
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0


def load_llm_config(
    provider: str | None = None, *, load_env: bool = True
) -> LLMConfig:
    """Load provider configuration from environment variables and optional .env."""
    if load_env:
        load_dotenv(override=False)

    raw_provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if raw_provider not in ("gemini", "openai", "ollama"):
        raise UnsupportedProviderError(
            f"Unsupported LLM provider '{raw_provider}'. "
            "Supported values: gemini, openai, ollama."
        )

    resolved: ProviderName = raw_provider  # type: ignore[assignment]
    if resolved == "gemini":
        return LLMConfig(
            provider="gemini",
            api_key=(os.getenv("GEMINI_API_KEY") or "").strip() or None,
            chat_model=(os.getenv("GEMINI_CHAT_MODEL") or "gemini-3.8-flash").strip(),
            embed_model=(
                os.getenv("GEMINI_EMBED_MODEL") or "gemini-embedding-001"
            ).strip(),
        )
    if resolved == "openai":
        return LLMConfig(
            provider="openai",
            api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
            chat_model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
            embed_model=(
                os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
            ).strip(),
        )
    return LLMConfig(
        provider="ollama",
        base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
        .strip()
        .rstrip("/"),
        chat_model=(os.getenv("OLLAMA_CHAT_MODEL") or "llama3.1:8b").strip(),
        embed_model=(os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text").strip(),
    )


def _build_chat_messages(
    prompt: str, system_prompt: str | None
) -> list[dict[str, str]]:
    """Build role-tagged chat messages for OpenAI and Ollama chat APIs."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_balanced_json(text: str) -> str | None:
    """Extract the first balanced JSON object or array substring from prose."""
    start = -1
    open_ch = ""
    close_ch = ""
    for idx, ch in enumerate(text):
        if ch == "{":
            start, open_ch, close_ch = idx, "{", "}"
            break
        if ch == "[":
            start, open_ch, close_ch = idx, "[", "]"
            break
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _build_json_candidates(raw_text: str) -> list[str]:
    """Generate ordered candidate JSON strings from raw model text."""
    stripped = raw_text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if fenced and fenced not in candidates:
            candidates.append(fenced)

    for source in list(candidates):
        balanced = _extract_balanced_json(source)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        no_trailing = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if no_trailing != candidate:
            expanded.append(no_trailing)
    return expanded


def parse_structured_response(raw_text: str, schema: type[T]) -> T:
    """Parse and validate a Pydantic model from raw JSON or fenced text."""
    candidates = _build_json_candidates(raw_text)
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            try:
                loaded = json.loads(candidate)
                return schema.model_validate(loaded)
            except (ValidationError, ValueError, TypeError) as inner_exc:
                last_error = inner_exc

    raise ValueError(f"Could not parse valid {schema.__name__}: {last_error}")


class BaseLLMAdapter(ABC):
    """Common interface and retry/repair wrapper for LLM and embedding providers."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        embed_batch_size: int = 64,
    ) -> None:
        self.config = config
        self.provider_name = config.provider
        self.chat_model = config.chat_model
        self.embed_model = config.embed_model
        self._sleep = sleep_fn
        self._embed_batch_size = max(1, embed_batch_size)

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate plain text with exponential backoff retries on transient errors."""
        return self._call_with_retry(
            lambda: self._raw_generate_text(
                prompt, system_prompt=system_prompt, temperature=temperature
            )
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate and validate a Pydantic model, repairing malformed JSON once."""
        parsed, raw_text = self._call_with_retry(
            lambda: self._raw_generate_structured(
                prompt, schema, system_prompt=system_prompt, temperature=temperature
            )
        )
        if parsed is not None:
            return schema.model_validate(parsed)

        try:
            return parse_structured_response(raw_text, schema)
        except ValueError as first_error:
            repair_prompt = (
                "The previous response did not match the required JSON schema.\n"
                f"Validation error: {first_error}\n"
                f"Required JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                f"Malformed response:\n{raw_text}\n\n"
                "Return ONLY valid JSON matching the schema, with no extra commentary."
            )
            repaired_text = self.generate_text(
                repair_prompt,
                system_prompt="You repair invalid JSON into valid JSON matching the schema.",
                temperature=0.0,
            )
            try:
                return parse_structured_response(repaired_text, schema)
            except ValueError as second_error:
                raise StructuredOutputError(
                    f"Failed to parse structured output for {schema.__name__} "
                    f"after repair: {second_error}"
                ) from second_error

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a list of strings."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for idx in range(0, len(texts), self._embed_batch_size):
            batch = texts[idx : idx + self._embed_batch_size]
            batch_vectors = self._call_with_retry(lambda b=batch: self._raw_embed_batch(b))
            vectors.extend(batch_vectors)
        return vectors

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                is_last = attempt == attempts - 1
                if is_last or not self._is_transient_error(exc):
                    raise
                delay = min(
                    16.0, self.config.retry_base_delay * (2**attempt)
                )
                self._sleep(delay)

    @abstractmethod
    def _is_transient_error(self, exc: Exception) -> bool:
        """Return True if the provider exception is a transient 429/503/timeout error."""

    @abstractmethod
    def _raw_generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> str:
        """Execute one provider-specific text generation call."""

    @abstractmethod
    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[T | None, str]:
        """Execute one provider-specific structured generation call."""

    @abstractmethod
    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Execute one provider-specific embedding batch call."""


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini API adapter backed by google-genai."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config, sleep_fn=sleep_fn)
        if client is not None:
            self._client = client
            return
        if not config.api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Add GEMINI_API_KEY=... to your .env "
                "file or environment before running with provider 'gemini'."
            )
        self._client = genai.Client(
            api_key=config.api_key,
            http_options=genai_types.HttpOptions(
                timeout=max(1000, int(config.timeout_seconds * 1000))
            ),
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, genai_errors.APIError):
            return exc.code in _TRANSIENT_HTTP_CODES
        return isinstance(exc, (TimeoutError, httpx.TimeoutException))

    def _raw_generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> str:
        cfg = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt,
        )
        response = self._client.models.generate_content(
            model=self.chat_model,
            contents=prompt,
            config=cfg,
        )
        return (response.text or "").strip()

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[T | None, str]:
        cfg = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._client.models.generate_content(
            model=self.chat_model,
            contents=prompt,
            config=cfg,
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed, (response.text or "")
        return None, (response.text or "")

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self.embed_model,
            contents=texts,
        )
        embeddings = getattr(response, "embeddings", None) or []
        return [list(item.values) for item in embeddings]


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI API adapter backed by the official openai SDK."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config, sleep_fn=sleep_fn)
        if client is not None:
            self._client = client
            return
        if not config.api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=... to your .env "
                "file or environment before running with provider 'openai'."
            )
        self._client = openai.OpenAI(
            api_key=config.api_key, timeout=config.timeout_seconds
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _TRANSIENT_HTTP_CODES
        return isinstance(exc, (TimeoutError, httpx.TimeoutException))

    def _raw_generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=_build_chat_messages(prompt, system_prompt),
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[T | None, str]:
        messages = _build_chat_messages(prompt, system_prompt)
        response = self._client.chat.completions.parse(
            model=self.chat_model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, schema):
            return parsed, (message.content or "")
        return None, (message.content or "")

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]


class OllamaAdapter(BaseLLMAdapter):
    """Local Ollama HTTP adapter backed by httpx."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.Client | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config, sleep_fn=sleep_fn)
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._client = http_client or httpx.Client(timeout=config.timeout_seconds)

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _TRANSIENT_HTTP_CODES
        return isinstance(exc, (httpx.TimeoutException, TimeoutError))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(f"{self._base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()

    def _raw_generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> str:
        data = self._post_json(
            "/api/chat",
            {
                "model": self.chat_model,
                "messages": _build_chat_messages(prompt, system_prompt),
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        return (data.get("message", {}).get("content") or "").strip()

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[T | None, str]:
        schema_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        data = self._post_json(
            "/api/chat",
            {
                "model": self.chat_model,
                "messages": _build_chat_messages(schema_prompt, system_prompt),
                "stream": False,
                "format": schema.model_json_schema(),
                "options": {"temperature": temperature},
            },
        )
        content = (data.get("message", {}).get("content") or "").strip()
        return None, content

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = self._post_json(
            "/api/embed",
            {
                "model": self.embed_model,
                "input": texts,
            },
        )
        return [list(vec) for vec in data.get("embeddings", [])]


def create_llm_adapter(
    provider: str | None = None,
    *,
    config: LLMConfig | None = None,
    load_env: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> BaseLLMAdapter:
    """Instantiate the configured LLM adapter from provider name or environment."""
    resolved = config or load_llm_config(provider, load_env=load_env)
    if resolved.provider == "gemini":
        return GeminiAdapter(resolved, sleep_fn=sleep_fn)
    if resolved.provider == "openai":
        return OpenAIAdapter(resolved, sleep_fn=sleep_fn)
    return OllamaAdapter(resolved, sleep_fn=sleep_fn)
