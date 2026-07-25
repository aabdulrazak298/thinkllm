from __future__ import annotations

import re
from collections.abc import AsyncIterator

from .cache import DebateCache
from .types import DebateConfig, DebateResult, Message, StreamEvent
from .agents import Agent

_CONVERGE_SIGNALS = [
    "agree", "acknowledged", "accepted", "converge", "concluded",
    "final strategy", "debate concluded", "settled", "resolved", "consensus",
    "converged", "we agree", "i accept", "i concur",
]
_DISAGREE_SIGNALS = [
    "counter:", "rebuttal:", "disagree", "wrong,", "flaw:", "you miss",
    "overlooked", "but you", "no, ", "incorrect", "false", "you fail",
    "violates", "not true",
]


def _has_converged(msg_a: str, msg_b: str, turn: int) -> bool:
    combined = (msg_a + " " + msg_b).lower()
    score = sum(1 for s in _CONVERGE_SIGNALS if s in combined)
    penalty = sum(1 for s in _DISAGREE_SIGNALS if s in combined)
    return score >= 2 and penalty == 0


class ThinkLLM:
    def __init__(self, config: DebateConfig, cache: DebateCache | None = None):
        self.config = config
        self.agent_a = Agent(config.debater_a)
        self.agent_b = Agent(config.debater_b)
        self.executor = Agent(config.executor)
        self._cache = cache

    async def run(self, query: str) -> DebateResult:
        transcript: list[Message] = [Message(role="user", content=query)]

        cached = self._load_cache(query)
        if cached is not None:
            transcript = cached
        else:
            for turn in range(self.config.max_turns):
                response_a = await self.agent_a.respond(transcript)
                transcript.append(Message(role="assistant", content=response_a, name=self.config.debater_a.name))

                response_b = await self.agent_b.respond(transcript)
                transcript.append(Message(role="assistant", content=response_b, name=self.config.debater_b.name))

                if self.config.early_termination and _has_converged(response_a, response_b, turn):
                    break

            self._save_cache(query, transcript)

        final_answer = await self._run_executor(query, transcript)

        return DebateResult(
            query=query,
            transcript=transcript,
            final_answer=final_answer,
        )

    async def stream(self, query: str) -> AsyncIterator[StreamEvent]:
        transcript: list[Message] = [Message(role="user", content=query)]
        max_turns = self.config.max_turns

        cached = self._load_cache(query)
        if cached is not None:
            for msg in cached[1:]:
                yield StreamEvent(
                    type="agent_message",
                    turn=0,
                    agent=msg.name or msg.role,
                    content=msg.content,
                )
            yield StreamEvent(type="converged", turn=max_turns)
        else:
            for turn in range(max_turns):
                yield StreamEvent(type="turn_start", turn=turn + 1)

                response_a = await self.agent_a.respond(transcript)
                transcript.append(Message(role="assistant", content=response_a, name=self.config.debater_a.name))
                yield StreamEvent(type="agent_message", turn=turn + 1, agent=self.config.debater_a.name, content=response_a)

                response_b = await self.agent_b.respond(transcript)
                transcript.append(Message(role="assistant", content=response_b, name=self.config.debater_b.name))
                yield StreamEvent(type="agent_message", turn=turn + 1, agent=self.config.debater_b.name, content=response_b)

                if self.config.early_termination and _has_converged(response_a, response_b, turn):
                    yield StreamEvent(type="converged", turn=turn + 1)
                    break

            self._save_cache(query, transcript)

        yield StreamEvent(type="executor_start")
        final_answer = await self._run_executor(query, transcript)
        yield StreamEvent(type="final_answer", content=final_answer)

    async def _run_executor(self, query: str, transcript: list[Message]) -> str:
        debate_text = "\n\n".join(
            f"[{m.name or m.role}]: {m.content}" for m in transcript[1:]
        )
        executor_messages = [
            Message(role="system", content=self.config.executor.system_prompt),
            Message(
                role="user",
                content=(
                    f"USER QUERY: {query}\n\n"
                    f"DEBATE TRANSCRIPT:\n{debate_text}\n\n"
                    f"Produce the final answer based on the debate strategy."
                ),
            ),
        ]
        return await self.executor.provider.generate(
            self.config.executor.model,
            executor_messages,
            **self.executor._generation_kwargs(),
        )

    def _load_cache(self, query: str) -> list[Message] | None:
        if self._cache is None:
            return None
        result = self._cache.get(query, self.config)
        if result is None:
            return None
        return result[0]

    def _save_cache(self, query: str, transcript: list[Message]) -> None:
        if self._cache is not None:
            self._cache.set(query, self.config, transcript)
