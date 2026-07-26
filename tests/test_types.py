import pytest
from pydantic import ValidationError

from thinkllm.types import (
    AgentConfig,
    DebateConfig,
    DebateResult,
    StreamEvent,
    StreamEventType,
)


class TestAgentConfig:
    def test_valid_config(self):
        cfg = AgentConfig(
            name="Test",
            model="gpt-4o",
            provider="openai",
            system_prompt="Be helpful.",
        )
        assert cfg.name == "Test"
        assert cfg.model == "gpt-4o"
        assert cfg.provider == "openai"
        assert cfg.system_prompt == "Be helpful."
        assert cfg.base_url is None
        assert cfg.temperature == 0.7
        assert cfg.top_p == 1.0
        assert cfg.max_context_tokens is None

    def test_with_base_url(self):
        cfg = AgentConfig(
            name="Test",
            model="deepseek-v4-flash",
            provider="openai",
            system_prompt="test",
            base_url="https://api.deepseek.com/v1",
        )
        assert cfg.base_url == "https://api.deepseek.com/v1"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(
                name="Test",
                model="gpt-4o",
                provider="invalid",
                system_prompt="test",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AgentConfig(name="Test", model="gpt-4o", provider="openai")

    def test_temperature_top_p_defaults(self):
        cfg = AgentConfig(
            name="Test",
            model="gpt-4o",
            provider="openai",
            system_prompt="test",
        )
        assert cfg.temperature == 0.7
        assert cfg.top_p == 1.0


class TestDebateConfig:
    def test_defaults(self):
        cfg = DebateConfig()
        assert cfg.max_turns == 3
        assert cfg.early_termination is True
        assert cfg.debater_a.name == "Debater A"
        assert cfg.debater_b.name == "Debater B"
        assert cfg.executor.name == "Executor"

    def test_custom_debaters(self):
        cfg = DebateConfig(
            max_turns=5,
            early_termination=False,
            debater_a=AgentConfig(
                name="Critic", model="gpt-4o", provider="openai", system_prompt="p"
            ),
            debater_b=AgentConfig(
                name="Builder", model="gpt-4o", provider="openai", system_prompt="p"
            ),
            executor=AgentConfig(
                name="Exec", model="gpt-4o", provider="openai", system_prompt="p"
            ),
        )
        assert cfg.max_turns == 5
        assert cfg.early_termination is False
        assert cfg.debater_a.name == "Critic"
        assert cfg.debater_b.name == "Builder"
        assert cfg.executor.name == "Exec"


class TestStreamEvent:
    def test_turn_start(self):
        ev = StreamEvent(type=StreamEventType.TURN_START, turn=1)
        assert ev.type == StreamEventType.TURN_START
        assert ev.turn == 1
        assert ev.agent is None
        assert ev.content is None

    def test_agent_message(self):
        ev = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            turn=2,
            agent="DebaterA",
            content="response",
        )
        assert ev.type == StreamEventType.AGENT_MESSAGE
        assert ev.turn == 2
        assert ev.agent == "DebaterA"
        assert ev.content == "response"

    def test_final_answer(self):
        ev = StreamEvent(
            type=StreamEventType.FINAL_ANSWER,
            content="the answer",
        )
        assert ev.type == StreamEventType.FINAL_ANSWER
        assert ev.turn is None
        assert ev.content == "the answer"

    def test_string_comparison_works(self):
        ev = StreamEvent(type=StreamEventType.TURN_START, turn=1)
        assert ev.type == "turn_start"


class TestDebateResult:
    def test_debate_result(self):
        dr = DebateResult(
            query="test query",
            transcript=[],
            final_answer="answer",
        )
        assert dr.query == "test query"
        assert dr.transcript == []
        assert dr.final_answer == "answer"
        assert dr.metadata == {}
