import pytest
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.function import FunctionModel

from thinkllm.agents import DebaterAgent, _message_text, estimate_message_tokens, estimate_tokens, trim_messages
from thinkllm.engine import ThinkLLM, _has_converged
from thinkllm.types import AgentConfig, StreamEventType

from .conftest import make_function_model


class TestConvergenceHeuristic:
    def test_convergence_with_signals(self):
        assert _has_converged("agree on factoring", "acknowledged, good point")

    def test_no_convergence_with_disagreement(self):
        assert not _has_converged("agree on X but you are wrong", "counter: no")

    def test_insufficient_signals(self):
        assert not _has_converged("maybe agree", "ok")


class TestAgent:
    @pytest.mark.asyncio
    async def test_respond_returns_model_output(self, agent_config):
        model = make_function_model(["test response"])
        agent = DebaterAgent(agent_config, _model=model)
        history = [ModelRequest(parts=[UserPromptPart(content="hello")])]
        result = await agent.respond(history)
        assert result == "test response"

    @pytest.mark.asyncio
    async def test_respond_passes_temperature_and_top_p(self, agent_config):
        agent_config.temperature = 0.5
        agent_config.top_p = 0.9
        model = make_function_model(["ok"])
        agent = DebaterAgent(agent_config, _model=model)
        settings = agent._agent.model_settings
        assert settings["temperature"] == 0.5
        assert settings["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_agent_respond_trims_when_over_limit(self, agent_config):
        agent_config.max_context_tokens = 50
        model = make_function_model(["ok"])
        agent = DebaterAgent(agent_config, _model=model)
        history = [
            ModelRequest(parts=[UserPromptPart(content="query")]),
            ModelResponse(parts=[TextPart(content="x " * 500)]),
        ]
        result = await agent.respond(history)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_agent_respond_no_trim_under_limit(self, agent_config):
        agent_config.max_context_tokens = 10000
        model = make_function_model(["ok"])
        agent = DebaterAgent(agent_config, _model=model)
        history = [
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="hello")]),
        ]
        result = await agent.respond(history)
        assert result == "ok"


class TestThinkLLM:
    @pytest.mark.asyncio
    async def test_run_debate_flow(self, debate_config):
        model = make_function_model(["A1 response", "B1 response", "A2 response", "B2 response", "Final answer"])
        engine = ThinkLLM(debate_config, _model=model)
        result = await engine.run("test query")

        assert result.query == "test query"
        assert result.final_answer == "Final answer"
        assert len(result.transcript) == 5
        assert _message_text(result.transcript[0]) == "test query"
        assert _message_text(result.transcript[1]) == "A1 response"
        assert _message_text(result.transcript[2]) == "B1 response"
        assert _message_text(result.transcript[3]) == "A2 response"
        assert _message_text(result.transcript[4]) == "B2 response"

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self, debate_config):
        debate_config.max_turns = 1
        model = make_function_model(["A1", "B1", "Final"])
        engine = ThinkLLM(debate_config, _model=model)
        result = await engine.run("query")

        assert len(result.transcript) == 3
        assert result.final_answer == "Final"

    @pytest.mark.asyncio
    async def test_early_termination(self, debate_config):
        debate_config.early_termination = True
        debate_config.max_turns = 3
        model = make_function_model(["agree, factoring is best", "acknowledged, we converge on this strategy", "Final"])
        engine = ThinkLLM(debate_config, _model=model)
        result = await engine.run("query")

        assert len(result.transcript) == 3
        assert result.final_answer == "Final"

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, debate_config):
        model = make_function_model(["A1", "B1", "A2", "B2", "Final"])
        engine = ThinkLLM(debate_config, _model=model)
        events = []
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
        model = make_function_model(["agree converge settle", "acknowledged accepted agreed", "Final"])
        engine = ThinkLLM(debate_config, _model=model)
        events = []
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

        model = make_function_model(["A1", "B1", "A2", "B2", "Final1", "Final2"])
        cache = DebateCache(tmp_path / "cache.db")
        engine = ThinkLLM(debate_config, cache=cache, _model=model)

        result1 = await engine.run("cached query")
        assert result1.final_answer == "Final1"
        assert len(result1.transcript) == 5

        result2 = await engine.run("cached query")
        assert result2.final_answer == "Final2"
        assert len(result2.transcript) == 5

        cache.close()

    @pytest.mark.asyncio
    async def test_stream_cache_hit(self, debate_config, tmp_path):
        from thinkllm.cache import DebateCache

        model = make_function_model(["A1", "B1", "A2", "B2", "Final1", "Final2"])
        cache = DebateCache(tmp_path / "cache.db")
        engine = ThinkLLM(debate_config, cache=cache, _model=model)

        await engine.run("stream-cached")
        events = []
        async for event in engine.stream("stream-cached"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_message" in event_types
        assert "converged" in event_types
        assert "executor_start" in event_types
        assert "final_answer" in event_types

        cache.close()


class TestContextWindow:
    def test_estimate_tokens(self):
        assert estimate_tokens("hello world") == 2
        assert estimate_tokens("") == 1

    def test_trim_drops_oldest_first(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="old msg " * 20)]),
            ModelResponse(parts=[TextPart(content="recent " * 10)]),
        ]
        result = trim_messages(messages, max_tokens=50)
        assert len(result) == 1
        assert "recent" in _message_text(result[0])

    def test_trim_within_limit_returns_all(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="there")]),
        ]
        result = trim_messages(messages, max_tokens=1000)
        assert len(result) == 2

    def test_estimate_message_tokens(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="world")]),
        ]
        tokens = estimate_message_tokens(messages)
        assert tokens == 2
