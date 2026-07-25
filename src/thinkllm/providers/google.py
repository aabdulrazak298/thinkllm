from __future__ import annotations

import os
from typing import Optional

from google import genai
from google.genai import types as genai_types

from .base import BaseProvider
from ..types import Message


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)

    async def generate(self, model: str, messages: list[Message], **kwargs) -> str:
        contents: list[genai_types.Content] = []

        for m in messages:
            role = "user" if m.role in ("user", "system") else "model"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=m.content)])
            )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            **kwargs,
        )
        return response.text
