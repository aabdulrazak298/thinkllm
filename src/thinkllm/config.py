from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import AgentConfig, DebateConfig


def load_config(path: str | Path, **overrides: Any) -> DebateConfig:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    debater_a = AgentConfig(**data["debater_a"])
    debater_b = AgentConfig(**data["debater_b"])
    executor = AgentConfig(**data["executor"])

    return DebateConfig(
        max_turns=data.get("max_turns", 3),
        debater_a=debater_a,
        debater_b=debater_b,
        executor=executor,
    )
