from __future__ import annotations

from .types import AgentConfig, Message
from .providers import get_provider


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.content) for m in messages)


def trim_messages(messages: list[Message], max_tokens: int) -> list[Message]:
    if not messages:
        return []
    if messages[0].role != "system":
        kept = _count_fit(messages, max_tokens)
        return messages[-kept:] if kept > 0 else []

    system = messages[0]
    rest = messages[1:]
    sys_tokens = estimate_tokens(system.content)
    available = max_tokens - sys_tokens
    if available <= 0:
        return [system]

    kept = _count_fit(rest, available)
    return [system] + (rest[-kept:] if kept > 0 else [])


def _count_fit(messages: list[Message], max_tokens: int) -> int:
    count = 0
    used = 0
    for m in reversed(messages):
        t = estimate_tokens(m.content)
        if used + t > max_tokens:
            break
        used += t
        count += 1
    return count


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        kwargs: dict[str, object] = {}
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        self.provider = get_provider(config.provider, **kwargs)

    def _generation_kwargs(self) -> dict[str, object]:
        return {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

    async def respond(self, history: list[Message]) -> str:
        messages = [Message(role="system", content=self.config.system_prompt)] + history
        limit = self.config.max_context_tokens
        if limit is not None and estimate_message_tokens(messages) > limit:
            messages = trim_messages(messages, limit)
        return await self.provider.generate(self.config.model, messages, **self._generation_kwargs())
