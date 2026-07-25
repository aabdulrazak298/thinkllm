from __future__ import annotations

from thinkllm.types import AgentConfig, Message


class TestTypes:
    def test_message_dataclass(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.name is None

    def test_message_with_name(self):
        msg = Message(role="assistant", content="hi", name="agent_a")
        assert msg.role == "assistant"
        assert msg.content == "hi"
        assert msg.name == "agent_a"

    def test_agent_config_dataclass(self):
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
        assert cfg.api_key is None

    def test_agent_config_with_api_key(self):
        cfg = AgentConfig(
            name="Test",
            model="gpt-4o",
            provider="openai",
            system_prompt="Be helpful.",
            api_key="sk-123",
        )
        assert cfg.api_key == "sk-123"
