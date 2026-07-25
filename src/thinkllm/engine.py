from __future__ import annotations

from .types import DebateConfig, DebateResult, Message
from .agents import Agent


class ThinkLLM:
    def __init__(self, config: DebateConfig):
        self.config = config
        self.agent_a = Agent(config.debater_a)
        self.agent_b = Agent(config.debater_b)
        self.executor = Agent(config.executor)

    async def run(self, query: str) -> DebateResult:
        transcript: list[Message] = [Message(role="user", content=query)]

        for turn in range(self.config.max_turns):
            response_a = await self.agent_a.respond(transcript)
            transcript.append(Message(role="assistant", content=response_a, name=self.config.debater_a.name))

            response_b = await self.agent_b.respond(transcript)
            transcript.append(Message(role="assistant", content=response_b, name=self.config.debater_b.name))

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

        final_answer = await self.executor.provider.generate(
            self.config.executor.model,
            executor_messages,
        )

        return DebateResult(
            query=query,
            transcript=transcript,
            final_answer=final_answer,
        )
