from __future__ import annotations

import os
from typing import Optional

from anthropic import AsyncAnthropic

from .base import BaseProvider
from ..types import Message


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=key)

    async def generate(self, model: str, messages: list[Message], **kwargs) -> str:
        system_msg: Optional[str] = None
        api_messages: list[dict] = []

        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                api_messages.append({"role": m.role, "content": m.content})

        kwargs_with_system = kwargs.copy()
        if system_msg:
            kwargs_with_system["system"] = system_msg

        response = await self.client.messages.create(
            model=model,
            messages=api_messages,
            max_tokens=kwargs_with_system.pop("max_tokens", 4096),
            **kwargs_with_system,
        )
        return response.content[0].text
