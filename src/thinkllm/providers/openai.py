from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI

from .base import BaseProvider
from ..types import Message


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key, base_url=base_url)

    async def generate(self, model: str, messages: list[Message], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        return response.choices[0].message.content
