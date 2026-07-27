"""Test debaters in isolation — runs debate only, prints full transcript."""
import asyncio, os, sys, time
from pathlib import Path
from dotenv import load_dotenv

PROJ = Path("/home/one/cloud/projects/thinkllm")
sys.path.insert(0, str(PROJ / "src"))

# Env
load_dotenv(PROJ / ".env")
for _le in [Path("/home/one/cloud/projects/lowcostllm/.env"),
            Path("/home/one/filebrowser/files/projects/lowcostllm/.env")]:
    if _le.exists(): load_dotenv(_le, override=True); break

from thinkllm.config import load_config
from thinkllm.engine import ThinkLLM
from thinkllm.agents import _message_text
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

QUERY = "what is the most recent movies released last 3 months"

async def main():
    cfg = load_config(str(PROJ / "config_openrouter.yaml"))

    # Set API keys on config
    key = os.environ.get("EXPENSIVE_API_KEY")
    cfg.debater_a.api_key = key
    cfg.debater_b.api_key = key
    cfg.executor.api_key = key  # needed even though we skip executor

    # Override for test: no cache
    engine = ThinkLLM(cfg, cache=None)

    # Run debate manually (same as _debate in server.py)
    tx = [ModelRequest(
        parts=[UserPromptPart(content=QUERY)],
        timestamp=None
    )]

    t0 = time.time()
    for turn in range(cfg.max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        ra = await engine.agent_a.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=ra)], timestamp=None))
        print(f"\n[{cfg.debater_a.name.upper()}]:")
        print(ra)

        rb = await engine.agent_b.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=rb)], timestamp=None))
        print(f"\n[{cfg.debater_b.name.upper()}]:")
        print(rb)

        from thinkllm.engine import _has_converged
        if cfg.early_termination and _has_converged(ra, rb):
            print("\n[CONVERGED — early termination]")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s, {len(tx)-1} messages")
    print(f"Total tokens: ~{sum(len(_message_text(m))//4 for m in tx)}")

if __name__ == "__main__":
    asyncio.run(main())
