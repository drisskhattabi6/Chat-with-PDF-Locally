import os
import time
from dataclasses import dataclass

import requests

from config import SETTINGS
from schemas import ChatResult
from utils import coerce_message_content


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    kind: str
    base_url: str
    env_key_name: str | None
    needs_api_key: bool
    default_model: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "Ollama": ProviderPreset(
        name="Ollama",
        kind="ollama",
        base_url=SETTINGS.ollama_base_url,
        env_key_name=None,
        needs_api_key=False,
        default_model="",
    ),
    "OpenAI": ProviderPreset(
        name="OpenAI",
        kind="openai_compatible",
        base_url="https://api.openai.com/v1",
        env_key_name="OPENAI_API_KEY",
        needs_api_key=True,
        default_model="",
    ),
    "Groq": ProviderPreset(
        name="Groq",
        kind="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        env_key_name="GROQ_API_KEY",
        needs_api_key=True,
        default_model="",
    ),
    "OpenRouter": ProviderPreset(
        name="OpenRouter",
        kind="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        env_key_name="OPENROUTER_API_KEY",
        needs_api_key=True,
        default_model="",
    ),
    "SambaNova": ProviderPreset(
        name="SambaNova",
        kind="openai_compatible",
        base_url="https://api.sambanova.ai/v1",
        env_key_name="SAMBANOVA_API_KEY",
        needs_api_key=True,
        default_model="",
    ),
}


def _normalize_openai_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _normalize_ollama_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api/chat"):
        base = base[: -len("/api/chat")]
    return base


def _parse_error(response: requests.Response) -> str:
    try:
        data = response.json()
    except Exception:
        return response.text.strip() or f"HTTP {response.status_code}"

    error = data.get("error")
    if isinstance(error, dict):
        return error.get("message", str(error))
    if error:
        return str(error)
    return response.text.strip() or f"HTTP {response.status_code}"


class OllamaChatClient:
    def __init__(self, base_url: str):
        self.provider_name = "Ollama"
        self.base_url = _normalize_ollama_base_url(base_url)

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=15)
        if not response.ok:
            raise RuntimeError(_parse_error(response))
        data = response.json()
        return sorted([item["name"] for item in data.get("models", []) if item.get("name")])

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> ChatResult:
        if not model:
            raise ValueError("Choose an Ollama model.")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        latency = time.perf_counter() - started

        if not response.ok:
            raise RuntimeError(_parse_error(response))

        data = response.json()
        message = data.get("message", {}) or {}
        text = coerce_message_content(message.get("content", ""))

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return ChatResult(
            text=text,
            provider="Ollama",
            model=model,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=data,
        )


class OpenAICompatibleChatClient:
    def __init__(self, provider_name: str, base_url: str, api_key: str):
        self.provider_name = provider_name
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.provider_name == "OpenRouter":
            headers["HTTP-Referer"] = SETTINGS.app_public_url
            headers["X-Title"] = SETTINGS.app_name

        return headers

    def list_models(self) -> list[str]:
        if not self.api_key:
            raise ValueError(f"{self.provider_name} API key is required.")

        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(_parse_error(response))

        data = response.json()
        raw_models = data.get("data") or data.get("models") or []

        model_ids: list[str] = []
        for item in raw_models:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("name") or item.get("model")
                if model_id:
                    model_ids.append(model_id)

        return sorted(set(model_ids))

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> ChatResult:
        if not self.api_key:
            raise ValueError(f"{self.provider_name} API key is required.")
        if not model:
            raise ValueError(f"Choose or enter a {self.provider_name} model.")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=300,
        )
        latency = time.perf_counter() - started

        if not response.ok:
            raise RuntimeError(_parse_error(response))

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"{self.provider_name} returned no choices.")

        message = choices[0].get("message", {}) or {}
        text = coerce_message_content(message.get("content", ""))

        usage = data.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return ChatResult(
            text=text,
            provider=self.provider_name,
            model=model,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=data,
        )


def build_chat_client(provider_name: str, base_url: str, api_key: str):
    preset = PROVIDER_PRESETS[provider_name]
    if preset.kind == "ollama":
        return OllamaChatClient(base_url=base_url or preset.base_url)
    return OpenAICompatibleChatClient(
        provider_name=provider_name,
        base_url=base_url or preset.base_url,
        api_key=api_key,
    )


def get_default_api_key(provider_name: str) -> str:
    preset = PROVIDER_PRESETS[provider_name]
    if not preset.env_key_name:
        return ""
    return os.getenv(preset.env_key_name, "")