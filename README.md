# thinkllm

> LLM agents debate strategy before answering — better answers through adversarial reasoning.

Two LLM agents debate **how** to answer your query (meta-strategists, not solvers), then an executor synthesizes the agreed strategy into a polished final answer. The debate is bot-to-bot using compressed, token-efficient language to minimize cost.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/providers-OpenAI%20%7C%20Anthropic%20%7C%20Google-orange" alt="Providers">
</p>

---

## How It Works

```
Query ──▶ Debater A (Critical Analyst)   ──▶ Debater B (Constructive Builder)
               │    ▲                              │    ▲
               ▼    │    3 turns each              ▼    │
            ┌───────┴────────┐              ┌───────┴────────┐
            │  Debate HOW    │─────────────▶│  Debate HOW    │
            │  to answer     │◀─────────────│  to answer     │
            └───────┬────────┘              └───────┬────────┘
                    │                               │
                    └───────────┬───────────────────┘
                                ▼
                         ┌─────────────┐
                         │  Executor   │
                         │  (Answer)   │
                         └──────┬──────┘
                                ▼
                          Final Answer
```

**Key insight:** Debaters are meta-strategists — they debate methodology (factoring vs quadratic formula, monolith vs microservices), never the answer itself. Only the executor sees the debate transcript and produces the human-facing output.

## Features

- **Multi-provider** — OpenAI, Anthropic, Google, or any OpenAI-compatible API (DeepSeek, Ollama, vLLM)
- **Compressed bot-to-bot debate** — debaters use shorthand/symbols to minimize token cost
- **Streaming output** — watch the debate unfold in real-time
- **Early termination** — stops debating when agents converge on a strategy
- **Retry logic** — exponential backoff on API failures (rate limits, timeouts, 5xx)
- **Two-tier caching** — in-memory LRU (1000 entries) + SQLite disk cache, skip repeated debates instantly
- **Context window management** — auto-trims old debate messages when approaching token limits
- **Interactive chat mode** — follow-up questions with conversation history
- **Temperature control** — per-agent temperature/top_p for creative debate vs deterministic execution
- **Config + CLI overrides** — YAML config with per-flag overrides

## Installation

```bash
git clone git@github.com:aabdulrazak298/thinkllm.git
cd thinkllm
python3 -m venv .venv
source .venv/bin/activate
pip install -e "."
```

## Quick Start

### 1. Set your API key

```bash
# Create .env in the project root
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 2. Run a query

```bash
thinkllm -q "Should I use React or Vue for my startup?"
```

### 3. See the debate

```bash
thinkllm -q "Explain garbage collection" --verbose
```

## Configuration

### DeepSeek / OpenAI-compatible

```yaml
# config_deepseek.yaml
max_turns: 3
early_termination: true

debater_a:
  name: "Critical Analyst"
  model: "deepseek-v4-flash"
  provider: "openai"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.8
  max_context_tokens: 4096
  system_prompt: |
    You are a META-STRATEGIST: debate methodology, not outcomes.
    CRITICAL: Do NOT compute the answer yourself. Only debate HOW to answer.

debater_b:
  name: "Constructive Builder"
  model: "deepseek-v4-flash"
  provider: "openai"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.8
  max_context_tokens: 4096
  system_prompt: |
    You are a META-STRATEGIST: debate methodology, not outcomes.
    CRITICAL: Do NOT compute the answer yourself. Only propose approaches.

executor:
  name: "Executor"
  model: "deepseek-v4-flash"
  provider: "openai"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.3
  max_context_tokens: 8192
  system_prompt: |
    You are the Executor. Apply the debated strategy to produce the final answer.
    CRITICAL: Do NOT mention the debate. Output only the standalone answer.
```

### Multiple providers

```yaml
debater_a:
  provider: "openai"
  model: "gpt-4o"

debater_b:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"

executor:
  provider: "google"
  model: "gemini-2.5-flash"
```

## CLI

```bash
# Basic query
thinkllm -q "Explain monads"

# Custom config
thinkllm -c config_deepseek.yaml -q "How do I learn Rust?"

# Streaming with debate transcript
thinkllm -c config_deepseek.yaml -q "Python vs Go for APIs" -v

# Force all 3 turns (no early termination)
thinkllm --no-early -q "Why use Docker?"

# Skip cache (always run fresh debate)
thinkllm --no-cache -q "Should I use microservices?"

# One-shot mode (no streaming)
thinkllm --no-stream -q "Explain the CAP theorem"

# Interactive chat mode
thinkllm --chat -v

# Model override from CLI
thinkllm --model gpt-4o --model-b claude-sonnet-4 --model-executor gpt-4o -q "..."
```

### Options

| Flag | Description |
|------|-------------|
| `-c, --config` | Path to config YAML (default: `config.yaml`) |
| `-q, --query` | The query to process |
| `-v, --verbose` | Show full debate transcript |
| `--model` | Override debater A model |
| `--model-b` | Override debater B model |
| `--model-executor` | Override executor model |
| `--no-stream` | Disable streaming output |
| `--no-early` | Disable early termination |
| `--no-cache` | Disable debate cache |
| `--chat` | Interactive chat mode |

## Python SDK

```python
import asyncio
from thinkllm import ThinkLLM, load_config

async def main():
    cfg = load_config("config.yaml")
    engine = ThinkLLM(cfg)

    # Simple run
    result = await engine.run("What is duck typing?")
    print(result.final_answer)

    # Streaming with events
    async for event in engine.stream("What is duck typing?"):
        if event.type == "turn_start":
            print(f"\n--- Turn {event.turn} ---")
        elif event.type == "agent_message":
            print(f"[{event.agent}]: {event.content}")
        elif event.type == "final_answer":
            print(f"\nAnswer: {event.content}")

asyncio.run(main())
```

### Programmatic config

```python
from thinkllm import ThinkLLM, AgentConfig, DebateConfig
from thinkllm.cache import DebateCache

config = DebateConfig(
    max_turns=3,
    early_termination=True,
    debater_a=AgentConfig(
        name="Critic",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a critical debater...",
        temperature=0.8,
    ),
    debater_b=AgentConfig(
        name="Builder",
        model="gpt-4o",
        provider="openai",
        system_prompt="You are a constructive debater...",
        temperature=0.8,
    ),
    executor=AgentConfig(
        name="Synthesizer",
        model="gpt-4o",
        provider="openai",
        system_prompt="Produce the final answer...",
        temperature=0.3,
    ),
)

cache = DebateCache(memory_cache_size=500)
engine = ThinkLLM(config, cache=cache)
result = await engine.run("My question")
cache.close()
```

## Architecture

```
src/thinkllm/
├── types.py         # Message, AgentConfig, DebateConfig, DebateResult, StreamEvent
├── providers/
│   ├── base.py      # Abstract BaseProvider
│   ├── openai.py    # OpenAI + any compatible API (DeepSeek, Ollama)
│   ├── anthropic.py # Anthropic Claude
│   └── google.py    # Google Gemini
├── agents.py        # Agent wrapper (system prompt + provider + context trimming)
├── engine.py        # Orchestrator: debate loop, convergence detection, streaming
├── cache.py         # Two-tier cache: in-memory LRU + SQLite disk
├── config.py        # YAML config loader with overrides
└── cli.py           # Click CLI (run, stream, chat modes)
```

## How It Works (Internals)

### Debate Protocol
- Agent A (Critical Analyst) and Agent B (Constructive Builder) alternate turns
- Each sees the full debate history (user query + all previous messages)
- System prompts instruct them to use compressed, symbolic language (bot-to-bot)
- They debate **approach/strategy**, never produce the final answer

### Early Termination
After each turn, the engine checks both messages for convergence signals:
```python
convergence = ("agree", "acknowledged", "converge", "consensus", ...)
disagreement = ("counter:", "rebuttal:", "disagree", "wrong", ...)
```
If both messages contain ≥2 convergence signals and zero disagreement → stop early.

### Context Window Management
When `max_context_tokens` is set, agent messages are trimmed before API calls:
- System prompt is always preserved
- Oldest messages are dropped first (FIFO)
- ~4 chars per token estimation (no external deps)

### Caching
- **Memory LRU**: 1000-entry `OrderedDict`, O(1) lookup, instant hits
- **Disk SQLite**: `~/.thinkllm/cache.db`, 64MB WAL mode, survives restarts
- Cache key: `SHA256(query) + SHA256(config fingerprint)` — changing personas/models invalidates cache

### Retry Logic
All providers wrapped with tenacity: 3 attempts, exponential backoff (1s→30s), retries on rate limits, timeouts, and server errors.

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a specific test
pytest tests/test_engine.py::TestThinkLLM::test_early_termination -v
```

## License

MIT
