import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from thinkllm.types import AgentConfig, DebateConfig


def make_function_model(responses: list[str]) -> FunctionModel:
    call_idx = 0

    async def fn(messages, info):
        nonlocal call_idx
        result = responses[call_idx % len(responses)]
        call_idx += 1
        return ModelResponse(parts=[TextPart(content=result)])

    return FunctionModel(function=fn)


@pytest.fixture
def agent_config():
    return AgentConfig(
        name="TestAgent",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def debate_config():
    return DebateConfig(
        max_turns=2,
        early_termination=False,
        debater_a=AgentConfig(
            name="DebaterA",
            model="gpt-4o",
            provider="openai",
            system_prompt="You are debater A.",
        ),
        debater_b=AgentConfig(
            name="DebaterB",
            model="gpt-4o",
            provider="openai",
            system_prompt="You are debater B.",
        ),
        executor=AgentConfig(
            name="Executor",
            model="gpt-4o",
            provider="openai",
            system_prompt="Synthesize.",
        ),
    )
