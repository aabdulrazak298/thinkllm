from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    TURN_START = "turn_start"
    AGENT_MESSAGE = "agent_message"
    CONVERGED = "converged"
    EXECUTOR_START = "executor_start"
    FINAL_ANSWER = "final_answer"


class AgentConfig(BaseModel):
    name: str
    model: str
    provider: Literal["openai", "anthropic", "google"]
    system_prompt: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_context_tokens: Optional[int] = None


class DebateConfig(BaseModel):
    max_turns: int = 3
    early_termination: bool = True
    debater_a: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="Debater A",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a debater.",
    ))
    debater_b: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="Debater B",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a debater.",
    ))
    executor: AgentConfig = Field(default_factory=lambda: AgentConfig(
        name="Executor",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a synthesis agent. Produce a clear final answer from the debate transcript.",
    ))


class DebateResult(BaseModel):
    query: str
    transcript: list  # list[ModelMessage] — imported lazily to avoid coupling
    final_answer: str
    metadata: dict = Field(default_factory=dict)


class StreamEvent(BaseModel):
    type: StreamEventType
    turn: Optional[int] = None
    agent: Optional[str] = None
    content: Optional[str] = None

