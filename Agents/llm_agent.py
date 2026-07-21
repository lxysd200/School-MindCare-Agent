from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterator, Sequence

from Agent_Type.AgentContext import AiMessage

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


LLMBackend = str
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
SUPPORTED_BACKENDS = {"openai", "ollama", "deepseek"}

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    backend: LLMBackend
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: float = 20.0
    temperature: float = 0.3


class LLMAgent:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any | None = None

    @property
    def config(self) -> LLMConfig:
        return self._config

    @classmethod
    def from_env(cls, agent_name: str | None = None) -> "LLMAgent":
        config = load_llm_config(agent_name=agent_name)
        return cls(config)

    def chat(self, messages: Sequence[dict[str, Any]]) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            messages=list(messages),
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is not installed.") from exc

        kwargs: dict[str, Any] = {
            "api_key": self._config.api_key,
            "timeout": self._config.timeout,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url

        self._client = OpenAI(**kwargs)
        return self._client

    def stream_chat(self, messages: Sequence[AiMessage]) -> Iterator[str]:
        client = self._get_client()
        normalized_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        stream = client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            messages=normalized_messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


def load_llm_config(agent_name: str | None = None) -> LLMConfig:
    backend = _resolve_backend(agent_name)
    timeout = _get_float_env(agent_name, "LLM_TIMEOUT", default=20.0)
    temperature = _get_float_env(agent_name, "LLM_TEMPERATURE", default=0.3)

    if backend == "ollama":
        return LLMConfig(
            backend=backend,
            model=_get_env(agent_name, "OLLAMA_MODEL") or "llama3",
            api_key=_get_env(agent_name, "OLLAMA_API_KEY") or "ollama",
            base_url=_get_env(agent_name, "OLLAMA_BASE_URL") or "http://localhost:11434/v1",
            timeout=timeout,
            temperature=temperature,
        )

    if backend == "deepseek":
        api_key = _get_env(agent_name, "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(_missing_key_message(agent_name, "DEEPSEEK_API_KEY"))
        return LLMConfig(
            backend=backend,
            model=_get_env(agent_name, "DEEPSEEK_MODEL", "OPENAI_MODEL") or DEFAULT_DEEPSEEK_MODEL,
            api_key=api_key,
            base_url=_get_env(agent_name, "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
            timeout=timeout,
            temperature=temperature,
        )

    api_key = _get_env(agent_name, "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_message(agent_name, "OPENAI_API_KEY"))
    return LLMConfig(
        backend=backend,
        model=_get_env(agent_name, "OPENAI_MODEL", "DEEPSEEK_MODEL") or "gpt-4o-mini",
        api_key=api_key,
        base_url=_get_env(agent_name, "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"),
        timeout=timeout,
        temperature=temperature,
    )


def has_llm_env(agent_name: str | None = None) -> bool:
    names = (
        "LLM_BACKEND",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "LLM_TIMEOUT",
        "LLM_TEMPERATURE",
    )
    return any(_get_env(agent_name, name) for name in names)


def _resolve_backend(agent_name: str | None) -> str:
    backend = (_get_env(agent_name, "LLM_BACKEND") or "").lower()
    if not backend:
        backend = "deepseek" if _has_deepseek_env(agent_name) else "openai"
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("LLM_BACKEND must be one of: openai, ollama, deepseek")
    return backend


def _get_env(agent_name: str | None, *names: str) -> str:
    agent_prefix = _normalize_agent_prefix(agent_name)
    for name in names:
        if agent_prefix:
            agent_value = (os.getenv(f"{agent_prefix}_{name}") or "").strip()
            if agent_value:
                return agent_value
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _get_float_env(agent_name: str | None, name: str, default: float) -> float:
    value = _get_env(agent_name, name)
    if not value:
        return default
    return float(value)


def _normalize_agent_prefix(agent_name: str | None) -> str:
    if not agent_name:
        return ""
    cleaned = agent_name.strip().upper()
    return cleaned.replace("-", "_").replace(" ", "_")


def _has_deepseek_env(agent_name: str | None = None) -> bool:
    return any(
        _get_env(agent_name, name)
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
    )


def _missing_key_message(agent_name: str | None, key_name: str) -> str:
    if not agent_name:
        return f"{key_name} is not configured."
    return f"{agent_name} agent requires {key_name} to be configured."
