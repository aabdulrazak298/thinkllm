from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


@dataclass
class AgentConfig:
    name: str
    model: str
    provider: str
    system_prompt: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0


@dataclass
class DebateConfig:
    max_turns: int = 3
    early_termination: bool = True
    debater_a: AgentConfig = field(default_factory=lambda: AgentConfig(
        name="Debater A",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a debater.",
    ))
    debater_b: AgentConfig = field(default_factory=lambda: AgentConfig(
        name="Debater B",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a debater.",
    ))
    executor: AgentConfig = field(default_factory=lambda: AgentConfig(
        name="Executor",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a synthesis agent. Produce a clear final answer from the debate transcript.",
    ))


@dataclass
class DebateResult:
    query: str
    transcript: list[Message]
    final_answer: str
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    type: str  # "turn_start", "agent_message", "converged", "executor_start", "final_answer"
    turn: Optional[int] = None
    agent: Optional[str] = None
    content: Optional[str] = None
