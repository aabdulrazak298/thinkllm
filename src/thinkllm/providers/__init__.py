from __future__ import annotations

from typing import Optional

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider

PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def get_provider(name: str, **kwargs: object) -> BaseProvider:
    provider_cls = PROVIDER_MAP.get(name)
    if provider_cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDER_MAP.keys())}")
    return provider_cls(**kwargs)


__all__ = ["BaseProvider", "OpenAIProvider", "AnthropicProvider", "GoogleProvider", "get_provider", "PROVIDER_MAP"]
