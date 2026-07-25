from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Message


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, model: str, messages: list[Message], **kwargs) -> str:
        ...
