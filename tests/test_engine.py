from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from thinkllm.types import AgentConfig, DebateConfig, Message, StreamEvent
from thinkllm.agents import Agent
from thinkllm.engine import ThinkLLM, _has_converged


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


class TestConvergenceHeuristic:
    def test_convergence_with_signals(self):
        assert _has_converged("agree on factoring", "acknowledged, good point", 0)

    def test_no_convergence_with_disagreement(self):
        assert not _has_converged("agree on X but you are wrong", "counter: no", 1)

    def test_insufficient_signals(self):
        assert not _has_converged("maybe agree", "ok", 1)


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

    @pytest.mark.asyncio
    async def test_respond_passes_temperature_and_top_p(self, agent_config):
        agent_config.temperature = 0.5
        agent_config.top_p = 0.9
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "ok"

        with patch("thinkllm.agents.get_provider", return_value=mock_provider):
            agent = Agent(agent_config)
            await agent.respond([Message(role="user", content="hi")])

            kwargs = mock_provider.generate.call_args[1]
            assert kwargs["temperature"] == 0.5
            assert kwargs["top_p"] == 0.9


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
    async def test_early_termination(self, debate_config):
        debate_config.early_termination = True
        debate_config.max_turns = 3
        mock = _make_mock_provider(["agree, factoring is best", "acknowledged, we converge on this strategy", "Final"])

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

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, debate_config):
        responses = ["A1", "B1", "A2", "B2", "Final"]
        mock = _make_mock_provider(responses)
        events: list[StreamEvent] = []

        with patch("thinkllm.agents.get_provider", return_value=mock):
            engine = ThinkLLM(debate_config)
            async for event in engine.stream("query"):
                events.append(event)

        event_types = [e.type for e in events]
        assert event_types == [
            "turn_start", "agent_message", "agent_message",
            "turn_start", "agent_message", "agent_message",
            "executor_start", "final_answer",
        ]
        assert events[-1].content == "Final"
        assert events[0].turn == 1
        assert events[3].turn == 2

    @pytest.mark.asyncio
    async def test_stream_early_termination(self, debate_config):
        debate_config.early_termination = True
        debate_config.max_turns = 3
        mock = _make_mock_provider(["agree converge settle", "acknowledged accepted agreed", "Final"])
        events: list[StreamEvent] = []

        with patch("thinkllm.agents.get_provider", return_value=mock):
            engine = ThinkLLM(debate_config)
            async for event in engine.stream("query"):
                events.append(event)

        event_types = [e.type for e in events]
        assert "converged" in event_types
        assert event_types == [
            "turn_start", "agent_message", "agent_message",
            "converged",
            "executor_start", "final_answer",
        ]

    @pytest.mark.asyncio
    async def test_cache_hit_skips_debate(self, debate_config, tmp_path):
        from thinkllm.cache import DebateCache

        debate_config.early_termination = False

        mock = _make_mock_provider(["A1", "B1", "A2", "B2", "Final1", "Final2"])

        with patch("thinkllm.agents.get_provider", return_value=mock):
            cache = DebateCache(tmp_path / "cache.db")
            engine = ThinkLLM(debate_config, cache=cache)

            result1 = await engine.run("cached query")
            assert result1.final_answer == "Final1"
            assert mock.generate.call_count == 5

            result2 = await engine.run("cached query")
            assert result2.final_answer == "Final2"
            assert mock.generate.call_count == 6

            cache.close()
