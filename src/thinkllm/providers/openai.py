from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseProvider
from ..types import Message


RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(self, model: str, messages: list[Message], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        return response.choices[0].message.content
