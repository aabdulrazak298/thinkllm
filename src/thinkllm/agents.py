from typing import Any

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .types import AgentConfig


def _message_text(msg: ModelMessage) -> str:
    parts = getattr(msg, "parts", [])
    return " ".join(getattr(p, "content", "") or "" for p in parts)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[ModelMessage]) -> int:
    return sum(estimate_tokens(_message_text(m)) for m in messages)


def trim_messages(messages: list[ModelMessage], max_tokens: int) -> list[ModelMessage]:
    if not messages:
        return []
    kept = _count_fit(messages, max_tokens)
    return messages[-kept:] if kept > 0 else []


def _count_fit(messages: list[ModelMessage], max_tokens: int) -> int:
    count = 0
    used = 0
    for m in reversed(messages):
        t = estimate_tokens(_message_text(m))
        if used + t > max_tokens:
            break
        used += t
        count += 1
    return count


class DebaterAgent:
    def __init__(self, config: AgentConfig, _model: Any = None):
        self.config = config
        self._model = _model or self._resolve_model(config)
        self._agent = PydanticAgent(
            self._model,
            system_prompt=config.system_prompt,
            model_settings=ModelSettings(
                temperature=config.temperature,
                top_p=config.top_p,
            ),
        )

    @staticmethod
    def _resolve_model(config: AgentConfig):
        if config.provider == "openai" and config.base_url:
            provider = OpenAIProvider(base_url=config.base_url)
            return OpenAIChatModel(config.model, provider=provider)
        return f"{config.provider}:{config.model}"

    async def respond(self, history: list[ModelMessage]) -> str:
        messages = list(history)
        limit = self.config.max_context_tokens
        if limit is not None and estimate_message_tokens(messages) > limit:
            messages = trim_messages(messages, limit)

        prompt = _message_text(messages[-1]) if messages else ""
        prior = messages[:-1] if len(messages) > 1 else messages

        result = await self._agent.run(
            user_prompt=prompt,
            message_history=prior,
        )
        return result.output
