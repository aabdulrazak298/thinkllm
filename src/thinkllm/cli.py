from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .config import load_config
from .engine import ThinkLLM


def _load_env() -> None:
    for path in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        env_file = path / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return
    load_dotenv()


@click.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML file")
@click.option("--query", "-q", default=None, help="The query to process")
@click.option("--model", default=None, help="Override model for debater A")
@click.option("--model-b", default=None, help="Override model for debater B")
@click.option("--model-executor", default=None, help="Override model for executor")
@click.option("--verbose", "-v", is_flag=True, help="Show full debate transcript")
def main(
    config: str,
    query: str | None,
    model: str | None,
    model_b: str | None,
    model_executor: str | None,
    verbose: bool,
) -> None:
    _load_env()

    if query is None:
        query = click.prompt("Enter your query")

    cfg = load_config(
        config,
        model=model,
        model_b=model_b,
        model_executor=model_executor,
        max_turns=3,
    )

    engine = ThinkLLM(cfg)

    click.echo("\nThinking...\n", err=True)

    result = asyncio.run(engine.run(query))

    if verbose:
        click.echo("=== DEBATE TRANSCRIPT ===\n")
        for msg in result.transcript:
            role_label = msg.name or msg.role
            click.echo(f"[{role_label}]: {msg.content}\n")

    click.echo(result.final_answer)


if __name__ == "__main__":
    main()
