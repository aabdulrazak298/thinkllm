import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from thinkllm.config import load_config
from thinkllm.types import DebateConfig


class TestLoadConfig:
    def test_load_basic_config(self):
        config_data = {
            "max_turns": 3,
            "debater_a": {
                "name": "Critic",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "You are a critic.",
            },
            "debater_b": {
                "name": "Builder",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "You are a builder.",
            },
            "executor": {
                "name": "Exec",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "Synthesize.",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            cfg = load_config(config_path)
            assert isinstance(cfg, DebateConfig)
            assert cfg.max_turns == 3
            assert cfg.debater_a.name == "Critic"
            assert cfg.debater_b.name == "Builder"
            assert cfg.executor.name == "Exec"
        finally:
            Path(config_path).unlink()

    def test_load_config_default_max_turns(self):
        config_data = {
            "debater_a": {
                "name": "A",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "debater_b": {
                "name": "B",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "executor": {
                "name": "E",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            cfg = load_config(config_path)
            assert cfg.max_turns == 3
        finally:
            Path(config_path).unlink()

    def test_load_config_overrides(self):
        config_data = {
            "debater_a": {
                "name": "A",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "debater_b": {
                "name": "B",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "executor": {
                "name": "E",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            cfg = load_config(config_path, max_turns=5)
            assert cfg.max_turns == 5
        finally:
            Path(config_path).unlink()

    def test_load_config_with_base_url(self):
        config_data = {
            "debater_a": {
                "name": "A",
                "model": "deepseek-v4-flash",
                "provider": "openai",
                "system_prompt": "prompt",
                "base_url": "https://api.deepseek.com/v1",
            },
            "debater_b": {
                "name": "B",
                "model": "deepseek-v4-flash",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "executor": {
                "name": "E",
                "model": "deepseek-v4-flash",
                "provider": "openai",
                "system_prompt": "prompt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            cfg = load_config(config_path)
            assert cfg.debater_a.base_url == "https://api.deepseek.com/v1"
        finally:
            Path(config_path).unlink()

    def test_load_config_invalid_provider_rejected(self):
        config_data = {
            "debater_a": {
                "name": "A",
                "model": "gpt-4o",
                "provider": "invalid-provider",
                "system_prompt": "prompt",
            },
            "debater_b": {
                "name": "B",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
            "executor": {
                "name": "E",
                "model": "gpt-4o",
                "provider": "openai",
                "system_prompt": "prompt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
