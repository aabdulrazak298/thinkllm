from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .cache import DebateCache
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
@click.option("--no-stream", is_flag=True, help="Disable streaming, show result all at once")
@click.option("--no-early", is_flag=True, help="Disable early termination")
@click.option("--no-cache", is_flag=True, help="Disable debate cache")
def main(
    config: str,
    query: str | None,
    model: str | None,
    model_b: str | None,
    model_executor: str | None,
    verbose: bool,
    no_stream: bool,
    no_early: bool,
    no_cache: bool,
) -> None:
    _load_env()

    if query is None:
        query = click.prompt("Enter your query")

    cfg = load_config(
        config,
        max_turns=3,
        early_termination=not no_early,
    )

    cache = None if no_cache else DebateCache()

    if model is not None:
        cfg.debater_a.model = model
    if model_b is not None:
        cfg.debater_b.model = model_b
    if model_executor is not None:
        cfg.executor.model = model_executor

    engine = ThinkLLM(cfg, cache=cache)

    if no_stream:
        click.echo("\nThinking...\n", err=True)
        result = asyncio.run(engine.run(query))
        if verbose:
            click.echo("=== DEBATE TRANSCRIPT ===\n")
            for msg in result.transcript:
                role_label = msg.name or msg.role
                click.echo(f"[{role_label}]: {msg.content}\n")
            click.echo("=== FINAL ANSWER (Executor) ===\n")
        click.echo(result.final_answer)
    else:
        asyncio.run(_stream_cli(engine, query, verbose))

    if cache is not None:
        cache.close()


async def _stream_cli(engine: ThinkLLM, query: str, verbose: bool) -> None:
    if verbose:
        click.echo("=== DEBATE TRANSCRIPT ===\n")

    async for event in engine.stream(query):
        if event.type == "turn_start":
            if verbose:
                click.echo(f"\n--- Turn {event.turn}/{engine.config.max_turns} ---\n")
            else:
                click.echo(f"\rTurn {event.turn}/{engine.config.max_turns}...", nl=False, err=True)
        elif event.type == "agent_message":
            if verbose:
                click.echo(f"[{event.agent}]: {event.content}\n")
        elif event.type == "converged":
            if verbose:
                click.echo(f"\n[Converged at turn {event.turn}]\n")
        elif event.type == "executor_start":
            if verbose:
                click.echo("\n=== FINAL ANSWER (Executor) ===\n")
            else:
                click.echo("\n", err=True)
        elif event.type == "final_answer":
            click.echo(event.content)


if __name__ == "__main__":
    main()
