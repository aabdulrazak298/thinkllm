# Plan: Convert to Pydantic AI

## Goal

Replace the custom provider abstraction (`src/thinkllm/providers/`) with [Pydantic AI](https://pydantic.dev/docs/ai/overview/), eliminating per-provider niche handling for message conversion, role remapping, response extraction, retry logic, and manual serialization.

## Architecture Overview

Pydantic AI provides a unified `Agent` abstraction over OpenAI, Anthropic, Google, and more. Its model strings (`"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-6"`, `"google:gemini-2.5-pro"`) handle all provider-specific conversion internally.

**Before:**
```
Agent.respond() → provider.generate(model, [system, history...]) → manual dict conversion → SDK → manual response extraction → str
```

**After:**
```
DebaterAgent.respond() → pydantic_ai Agent.run(prompt, message_history=transcript) → str
```

The `message_history` parameter carries the full debate transcript as `list[ModelMessage]`. Each agent's `system_prompt` is applied per-run automatically via the `instructions=` parameter.

## Files

### Removed (5 files)
- `src/thinkllm/providers/__init__.py` — no more provider registry
- `src/thinkllm/providers/base.py` — no more custom base class
- `src/thinkllm/providers/openai.py` — Pydantic AI handles OpenAI
- `src/thinkllm/providers/anthropic.py` — Pydantic AI handles Anthropic
- `src/thinkllm/providers/google.py` — Pydantic AI handles Google

### Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | Replace `openai`, `anthropic`, `google-genai` with `pydantic-ai>=0.0.50`. Remove `tenacity` (currently an undeclared transitive dependency from `anthropic` — Pydantic AI manages retries internally). |
| `src/thinkllm/types.py` | `@dataclass` → Pydantic `BaseModel` for `AgentConfig`, `DebateConfig`, `DebateResult`, `StreamEvent`. Drop `Message` type (use Pydantic AI's `ModelMessage`). Provider field uses `Literal["openai","anthropic","google"]`. Add `StreamEventType` enum. |
| `src/thinkllm/agents.py` | Replace `Agent` class with `DebaterAgent` wrapping Pydantic AI's `Agent`. Uses `message_history` for transcript. Context trimming via `ProcessHistory` capability. Removes `get_provider` dependency. **The executor will also use `DebaterAgent` instead of calling `provider.generate()` directly** — currently `_run_executor()` bypasses `Agent.respond()` entirely, which skips context trimming and creates an inconsistent code path. |
| `src/thinkllm/engine.py` | Transcript uses Pydantic AI `ModelMessage` types instead of custom `Message`. Streaming stays at **turn/event granularity** (not token-level — see Risks). Debate loop logic preserved. Remove dead `turn` parameter from `_has_converged()` (currently accepted but never used). |
| `src/thinkllm/config.py` | `AgentConfig.model_validate()` replaces bare `**` unpack. Provider name validated by Pydantic. `api_key` removed; `base_url` preserved as `Optional[str]` for OpenAI-compatible providers (see DeepSeek migration below). |
| `src/thinkllm/cache.py` | `ModelMessagesTypeAdapter` for serialization; replaces manual JSON dict/list comprehensions. **Fix cache fingerprint** to include `early_termination`, temperature, top_p, and full executor config (currently missing — changing executor model does not invalidate cache). |
| `src/thinkllm/cli.py` | Adapt to new types; same Click interface. Agent name attribution tracked externally instead of on `Message.name`. **Add `tests/test_cli.py`** with smoke tests for `main()`, `_stream_cli()`, and `_chat_loop()` (currently 0% CLI test coverage). |
| `config.yaml` / `config_deepseek.yaml` | See config migration section below for exact format changes. |

### Deleted test files (rewritten)
- `tests/test_types.py` → Rewritten as Pydantic model validation tests
- `tests/test_engine.py` → Replace `AsyncMock` provider patches with Pydantic AI's `FunctionModel` for unit testing. Add `conftest.py` with shared fixtures.
- `tests/test_config.py` → Update to Pydantic model construction; add validation tests for error paths (missing keys, invalid YAML, unknown providers)
- **New:** `tests/test_cli.py` → Smoke/integration tests for CLI entry point

## Config YAML Migration

### Before (current format):
```yaml
debater_a:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-..."       # removed
  base_url: "https://..."  # preserved for OpenAI-compatible
  temperature: 0.8
  system_prompt: "..."
```

### After (new format):
```yaml
debater_a:
  provider: "openai"
  model: "gpt-4o"
  base_url: null           # optional, only used when provider="openai" and using compatible API
  temperature: 0.8
  system_prompt: "..."
```

### DeepSeek config migration:
```yaml
# Before
debater_a:
  provider: "openai"
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com/v1"

# After — same, base_url is preserved
debater_a:
  provider: "openai"
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com/v1"
```

Pydantic AI models are constructed with `OpenAIModel(model_name, base_url=...)` when `base_url` is set, or use the standard env-var-based model when it's not. The `provider` field stays as a simple `"openai" | "anthropic" | "google"` key; the full Pydantic AI model string (`"openai:gpt-4o"`) is assembled internally by `DebaterAgent`.

## What gets eliminated

1. **~150 lines of provider code** (3 provider files + base + registry)
2. **Manual role remapping** (Google: `user`/`system`→`user`, `assistant`→`model`; Anthropic: system message extraction from message list)
3. **Manual response extraction** (`response.choices[0].message.content` / `response.content[0].text` / `response.text`)
4. **Manual retry decorators** (3 copies of `@retry` with provider-specific exception types)
5. **`api_key` field in `AgentConfig`** (Pydantic AI uses env vars; for per-agent overrides, users construct models explicitly with `OpenAIModel(api_key=...)`). `base_url` is kept.
6. **Manual `**dict` unpack in config loading** (Pydantic validation handles this)
7. **Manual JSON serialization in cache** (`json.dumps` comprehensions → `ModelMessagesTypeAdapter`)
8. **Custom `Message` type** (replaced by Pydantic AI's `ModelMessage` union type)
9. **Undeclared `tenacity` dependency** (latent bug: used in all 3 providers but never listed in `pyproject.toml`, pulled in transitively via `anthropic`)
10. **Dead `turn` parameter** in `_has_converged()` — accepted but never used

---

## Step-by-step implementation order

### Phase 1: Foundation (commits 1-2)
1. **`pyproject.toml`** — Add `pydantic-ai>=0.0.50` dependency; remove `openai`, `anthropic`, `google-genai`; remove `tenacity` (undeclared)
2. **`src/thinkllm/types.py`** — Convert dataclasses to Pydantic `BaseModel`. Drop `Message`. Add `StreamEventType` enum. Keep `base_url` on `AgentConfig`, remove `api_key`.

### Phase 2: Core (commits 3-5)
3. **`src/thinkllm/agents.py`** — Replace `Agent` with `DebaterAgent` wrapping Pydantic AI's `Agent`. Context trimming via `ProcessHistory`. Executor path unified (no more direct `provider.generate()` call).
4. **`src/thinkllm/engine.py`** — Adapt to `ModelMessage`. Keep streaming at turn granularity. Remove dead `turn` param from `_has_converged()`. Delegate executor call through `DebaterAgent` instead of `_run_executor()` calling provider directly.
5. **`src/thinkllm/config.py`** — Use `model_validate()`. Remove `api_key` handling. Preserve `base_url` passthrough.

### Phase 3: Supporting (commits 6-8)
6. **`src/thinkllm/cache.py`** — `ModelMessagesTypeAdapter` for serialization. Fix config fingerprint to include `early_termination`, temperature, top_p, executor config.
7. **`src/thinkllm/cli.py`** — Adapt to new types. Track agent name externally. Fix Anthropic system-prompt-in-messages issue (see Risks).
8. **`src/thinkllm/__init__.py`** — Update exports. Remove provider re-exports.

### Phase 4: Cleanup & test (commits 9-12)
9. **Remove `src/thinkllm/providers/`** — Delete all 5 files.
10. **Update tests** — Rewrite `test_types.py`, `test_engine.py`, `test_config.py`. Add `conftest.py` with shared fixtures. Add `test_cli.py`.
11. **Update config YAML files** — Apply new format to `config.yaml` and `config_deepseek.yaml` (remove `api_key` fields, keep `base_url`).
12. **Run tests to verify** — `pytest tests/ -v`

### Migration strategy
- Create an isolated git worktree: `git worktree add ../thinkllm-pydantic main`
- Each commit above should leave the project in a runnable state
- Run `pytest tests/ -v` after every commit
- Only merge back to `main` when all tests pass

---

## Risks & considerations

1. **`base_url` must be preserved for DeepSeek/Ollama users.** Pydantic AI supports `OpenAIModel(model_name, base_url=...)` for custom endpoints. The `base_url` field stays on `AgentConfig` and is passed to `OpenAIModel` constructor when provider is `"openai"` and `base_url` is set. Without this, the `config_deepseek.yaml` use case breaks entirely.

2. **Streaming stays at turn granularity, not token-level.** Pydantic AI's `run_stream()` with `stream_text()` produces incremental token deltas. The current `stream()` yields complete `StreamEvent` objects per turn/agent. **We keep turn-level streaming** — each `agent_message` event still carries a full response string (we collect tokens via `run_stream()` then yield one event). If token-level streaming is desired later, it requires new `StreamEvent` types and CLI changes, but that is out of scope for this migration.

3. **Executor must use `DebaterAgent` too.** Currently `_run_executor()` (engine.py:100-119) calls `provider.generate()` directly, skipping context trimming and the unified `Agent.respond()` path. With Pydantic AI, the executor gets its own `DebaterAgent` and calls `agent.run()` — this is a design cleanup, not just a provider swap.

4. **Agent name attribution.** Pydantic AI `ModelResponse` has no `name` field. Track agent name externally in the debate loop (a dict mapping message index → agent name, or a parallel list). The CLI and executor transcript formatting read from this external mapping instead of `Message.name`.

5. **Cache fingerprint fix is in scope.** The current fingerprint excludes `early_termination`, temperature, top_p, and executor config — meaning you can change your executor model without invalidating cached debates. This is a bug. Fix it by adding these fields to the fingerprint string.

6. **Behavior change: retry logic.** Pydantic AI's internal retry behavior may differ from current tenacity config (3 retries, exp backoff 1s-30s). Test with a real API call before merging.

7. **Anthropic system prompt in `trim_messages()`.** The current `trim_messages()` preserves `messages[0]` if its role is `"system"` (agents.py:18). With Pydantic AI, system prompts are passed via `instructions=`, not as messages — so this preservation logic becomes unnecessary. `trim_messages()` should be updated to stop special-casing system messages.

8. **API key handling.** Pydantic AI reads keys from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Per-agent `api_key` overrides from config YAML are no longer supported. Users who need per-agent keys must set them via environment or construct `OpenAIModel(api_key=...)` explicitly in code (SDK-only use case).

9. **No token-level dependency for estimation.** The current `estimate_tokens()` uses `len(text) // 4` heuristic. Pydantic AI may provide token-counting utilities — if so, replace the heuristic. If not, keep it.

10. **Convergence heuristic brittleness.** `_has_converged()` uses substring matching on lowercase text — converge signals like `"agree"` match `"disagree"` because `"agree"` is a substring of `"disagree"`. This pre-existing issue is unchanged by the migration but is worth noting.
