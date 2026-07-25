from __future__ import annotations

from .types import AgentConfig, Message
from .providers import get_provider


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        kwargs: dict[str, object] = {}
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        self.provider = get_provider(config.provider, **kwargs)

    async def respond(self, history: list[Message]) -> str:
        messages = [Message(role="system", content=self.config.system_prompt)] + history
        return await self.provider.generate(self.config.model, messages)
