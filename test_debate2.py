"""Test debaters with tools injected into USER MESSAGE, not system prompt."""
import asyncio, os, sys, time
from pathlib import Path
from dotenv import load_dotenv

PROJ = Path("/home/one/cloud/projects/thinkllm")
sys.path.insert(0, str(PROJ / "src"))

load_dotenv(PROJ / ".env")
for _le in [Path("/home/one/cloud/projects/lowcostllm/.env"),
            Path("/home/one/filebrowser/files/projects/lowcostllm/.env")]:
    if _le.exists(): load_dotenv(_le, override=True); break

from thinkllm.config import load_config
from thinkllm.engine import ThinkLLM
from thinkllm.agents import _message_text
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from thinkllm.types import AgentConfig

USER_QUERY = "what is the most recent movies released last 3 months"

# Tools injected into the USER MESSAGE — not system prompt
DEBATE_QUERY = (
    "I am an AI that needs to answer this user question:\n\n"
    f"\"{USER_QUERY}\"\n\n"
    "I have access to these tools:\n"
    "• web_search(query) — live internet search (Google + DuckDuckGo)\n"
    "• web_fetch(url) — extract full text from any webpage\n"
    "• youtube_transcript(url) — get video subtitles\n"
    "• run_code(code) — Python sandbox (numpy, pandas, matplotlib)\n"
    "• generate_graph(x, y, type, title) — create charts\n"
    "• call_n8n(name, payload) — external API calls\n\n"
    "I need you two strategists to debate the BEST approach for answering this.\n"
    "Which tools should I use, in what order? What are the edge cases?\n"
    "IMPORTANT: I CAN use web_search. I am NOT limited to training data.\n"
    "Do NOT suggest I redirect the user — I have tools to answer directly.\n"
    "End with a concrete step-by-step plan."
)

# Minimal system prompts — just identity, no tools
MINIMAL_A = (
    "You are the CRITICAL ANALYST — a debate strategist. "
    "You poke holes, find edge cases, expose weak assumptions. "
    "You never answer the user. You debate strategy only. "
    "Be concise. Max 3 turns."
)

MINIMAL_B = (
    "You are the CONSTRUCTIVE BUILDER — a debate strategist. "
    "You refine approaches, build on good ideas, propose plans. "
    "You never answer the user. You debate strategy only. "
    "Counter defeatism — if the Analyst says something can't be done, push back. "
    "Be concise. Max 3 turns."
)

async def main():
    cfg = load_config(str(PROJ / "config_openrouter.yaml"))
    key = os.environ.get("EXPENSIVE_API_KEY")

    # Override system prompts with minimal versions
    cfg.debater_a.system_prompt = MINIMAL_A
    cfg.debater_b.system_prompt = MINIMAL_B
    cfg.debater_a.api_key = key
    cfg.debater_b.api_key = key
    cfg.executor.api_key = key

    engine = ThinkLLM(cfg, cache=None)

    # User message IS the debate query (with tools embedded)
    tx = [ModelRequest(parts=[UserPromptPart(content=DEBATE_QUERY)], timestamp=None)]

    t0 = time.time()
    for turn in range(cfg.max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}")
        print(f"{'='*60}")

        ra = await engine.agent_a.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=ra)], timestamp=None))
        print(f"\n[CRITICAL ANALYST]:")
        print(ra)

        rb = await engine.agent_b.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=rb)], timestamp=None))
        print(f"\n[CONSTRUCTIVE BUILDER]:")
        print(rb)

        from thinkllm.engine import _has_converged
        if cfg.early_termination and _has_converged(ra, rb):
            print("\n[CONVERGED]")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{elapsed:.1f}s | {len(tx)-1} msgs | ~{sum(len(_message_text(m))//4 for m in tx)} tokens")

asyncio.run(main())
