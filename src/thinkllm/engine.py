from collections.abc import AsyncIterator
from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from .agents import DebaterAgent, _message_text
from .cache import DebateCache
from .types import DebateConfig, DebateResult, StreamEvent, StreamEventType

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


def _has_converged(msg_a: str, msg_b: str) -> bool:
    combined = (msg_a + " " + msg_b).lower()
    score = sum(1 for s in _CONVERGE_SIGNALS if s in combined)
    penalty = sum(1 for s in _DISAGREE_SIGNALS if s in combined)
    return score >= 2 and penalty == 0


def _build_user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _build_response_message(content: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=content)],
        timestamp=datetime.now(timezone.utc),
    )


class ThinkLLM:
    def __init__(self, config: DebateConfig, cache: DebateCache | None = None, _model=None):
        self.config = config
        self.agent_a = DebaterAgent(config.debater_a, _model=_model, disable_thinking=True)
        self.agent_b = DebaterAgent(config.debater_b, _model=_model, disable_thinking=True)
        self.executor = DebaterAgent(config.executor, _model=_model)
        self._cache = cache

    async def run(self, query: str, primer: str | None = None) -> DebateResult:
        transcript = [_build_user_message(query)]
        if primer:
            transcript.append(_build_response_message(primer))

        cached = self._load_cache(query)
        if cached is not None:
            transcript = cached
        else:
            for turn in range(self.config.max_turns):
                response_a = await self.agent_a.respond(transcript)
                transcript.append(_build_response_message(response_a))

                response_b = await self.agent_b.respond(transcript)
                transcript.append(_build_response_message(response_b))

                if self.config.early_termination and _has_converged(response_a, response_b):
                    break

            self._save_cache(query, transcript)

        final_answer = await self._run_executor(query, transcript)

        return DebateResult(
            query=query,
            transcript=transcript,
            final_answer=final_answer,
        )

    async def stream(self, query: str, primer: str | None = None) -> AsyncIterator[StreamEvent]:
        transcript = [_build_user_message(query)]
        if primer:
            transcript.append(_build_response_message(primer))
        max_turns = self.config.max_turns

        cached = self._load_cache(query)
        if cached is not None:
            cached_transcript = cached
            agent_names = self._reconstruct_names(cached_transcript)
            for i in range(1, len(cached_transcript)):
                yield StreamEvent(
                    type=StreamEventType.AGENT_MESSAGE,
                    turn=0,
                    agent=agent_names[i] or "assistant",
                    content=_message_text(cached_transcript[i]),
                )
            yield StreamEvent(type=StreamEventType.CONVERGED, turn=max_turns)
            transcript = cached_transcript
        else:
            for turn in range(max_turns):
                yield StreamEvent(type=StreamEventType.TURN_START, turn=turn + 1)

                response_a = await self.agent_a.respond(transcript)
                transcript.append(_build_response_message(response_a))
                yield StreamEvent(
                    type=StreamEventType.AGENT_MESSAGE,
                    turn=turn + 1,
                    agent=self.config.debater_a.name,
                    content=response_a,
                )

                response_b = await self.agent_b.respond(transcript)
                transcript.append(_build_response_message(response_b))
                yield StreamEvent(
                    type=StreamEventType.AGENT_MESSAGE,
                    turn=turn + 1,
                    agent=self.config.debater_b.name,
                    content=response_b,
                )

                if self.config.early_termination and _has_converged(response_a, response_b):
                    yield StreamEvent(type=StreamEventType.CONVERGED, turn=turn + 1)
                    break

            self._save_cache(query, transcript)

        yield StreamEvent(type=StreamEventType.EXECUTOR_START)
        final_answer = await self._run_executor(query, transcript)
        yield StreamEvent(type=StreamEventType.FINAL_ANSWER, content=final_answer)

    async def _run_executor(self, query: str, transcript: list) -> str:
        agent_names = self._reconstruct_names(transcript)
        debate_text = "\n\n".join(
            f"[{agent_names[i] or 'unknown'}]: {_message_text(m)}"
            for i, m in enumerate(transcript[1:], start=1)
        )
        prompt = (
            f"USER QUERY: {query}\n\n"
            f"DEBATE TRANSCRIPT:\n{debate_text}\n\n"
            f"Produce the final answer based on the debate strategy."
        )
        return await self.executor.respond([_build_user_message(prompt)])

    def _reconstruct_names(self, transcript: list) -> list[str | None]:
        names: list[str | None] = [None]
        for i in range(1, len(transcript)):
            names.append(
                self.config.debater_a.name
                if i % 2 == 1
                else self.config.debater_b.name
            )
        return names

    def _load_cache(self, query: str) -> list | None:
        if self._cache is None:
            return None
        result = self._cache.get(query, self.config)
        if result is None:
            return None
        return result[0]

    def _save_cache(self, query: str, transcript: list) -> None:
        if self._cache is not None:
            self._cache.set(query, self.config, transcript)



