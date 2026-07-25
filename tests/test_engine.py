from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from thinkllm.types import AgentConfig, DebateConfig, Message
from thinkllm.agents import Agent
from thinkllm.engine import ThinkLLM


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


def _make_mock_provider(responses: list[str]):
    mock = AsyncMock()
    call_idx = 0

    async def side_effect(model, messages, **kwargs):
        nonlocal call_idx
        result = responses[call_idx % len(responses)]
        call_idx += 1
        return result

    mock.generate.side_effect = side_effect
    return mock


class TestAgent:
    @pytest.mark.asyncio
    async def test_respond_includes_system_prompt(self, agent_config):
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "test response"

        with patch("thinkllm.agents.get_provider", return_value=mock_provider):
            agent = Agent(agent_config)
            history = [Message(role="user", content="hello")]
            result = await agent.respond(history)

            assert result == "test response"
            call_messages = mock_provider.generate.call_args[0][1]
            assert call_messages[0].role == "system"
            assert call_messages[0].content == "You are a test agent."
            assert call_messages[1].role == "user"
            assert call_messages[1].content == "hello"


class TestThinkLLM:
    @pytest.mark.asyncio
    async def test_run_debate_flow(self, debate_config):
        responses = ["A1 response", "B1 response", "A2 response", "B2 response", "Final answer"]
        mock = _make_mock_provider(responses)

        with patch("thinkllm.agents.get_provider", return_value=mock):
            engine = ThinkLLM(debate_config)
            result = await engine.run("test query")

            assert result.query == "test query"
            assert result.final_answer == "Final answer"
            assert len(result.transcript) == 5
            assert result.transcript[0].content == "test query"
            assert result.transcript[1].content == "A1 response"
            assert result.transcript[2].content == "B1 response"
            assert result.transcript[3].content == "A2 response"
            assert result.transcript[4].content == "B2 response"
            assert mock.generate.call_count == 5

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self, debate_config):
        debate_config.max_turns = 1
        mock = _make_mock_provider(["A1", "B1", "Final"])

        with patch("thinkllm.agents.get_provider", return_value=mock):
            engine = ThinkLLM(debate_config)
            result = await engine.run("query")

            assert len(result.transcript) == 3
            assert result.final_answer == "Final"
            assert mock.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_run_executor_receives_debate_and_query(self, debate_config):
        responses = ["A1", "B1", "A2", "B2", "Final"]
        mock = _make_mock_provider(responses)

        with patch("thinkllm.agents.get_provider", return_value=mock):
            engine = ThinkLLM(debate_config)
            _ = await engine.run("my query")

            executor_call = mock.generate.call_args_list[4]
            executor_messages = executor_call[0][1]
            assert executor_messages[0].role == "system"
            assert executor_messages[0].content == "Synthesize."
            assert "USER QUERY: my query" in executor_messages[1].content
            assert "DEBATE TRANSCRIPT:" in executor_messages[1].content
            assert "[DebaterA]: A1" in executor_messages[1].content
            assert "[DebaterB]: B1" in executor_messages[1].content
