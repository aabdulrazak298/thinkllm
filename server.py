"""
ThinkLLM Webhook Server — Pydantic AI-powered.
Debate via thinkllm library, executor via pydantic-ai agent.
"""

from __future__ import annotations
import asyncio, json, os, re, sqlite3, subprocess, sys, time, urllib.parse
from pathlib import Path
from dotenv import load_dotenv

# --- Env ---
load_dotenv(Path(__file__).parent / ".env")
for _le in [Path("/home/one/cloud/projects/lowcostllm/.env"),
            Path("/home/one/filebrowser/files/projects/lowcostllm/.env")]:
    if _le.exists(): load_dotenv(_le, override=True); break
if not os.environ.get("OPENAI_API_KEY"):
    ds = os.environ.get("EXPENSIVE_API_KEY", "")
    ch = os.environ.get("CHEAP_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = ds or ch or ""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from thinkllm.cache import DebateCache
from thinkllm.engine import ThinkLLM, _has_converged
from thinkllm.config import load_config
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from datetime import datetime, timezone

# --- Config ---
CONFIG_PATH = Path(__file__).parent / "config_openrouter.yaml"
HTTP_PORT = int(os.environ.get("THINKLLM_PORT", "8802"))
HTTP_HOST = os.environ.get("THINKLLM_HOST", "0.0.0.0")
TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "FlaskWebHook2026!")
TG_CHAR_LIMIT = 4000
HISTORY_DB = Path(__file__).parent / "tg_history.db"
TRANSACTIONS_DB = Path(__file__).parent / "transactions.db"
TRANSACTION_LIMIT = 50  # keep last N full interactions

# Models registry (name → [model_id, base_url, key_env])
MODELS = {
    "pro":       ("deepseek-v4-pro",   "https://api.deepseek.com/v1",     "EXPENSIVE_API_KEY"),
    "flash":     ("deepseek-v4-flash",  "https://api.deepseek.com/v1",     "EXPENSIVE_API_KEY"),
    "r1":        ("deepseek-reasoner",  "https://api.deepseek.com/v1",     "EXPENSIVE_API_KEY"),
    "qwen":      ("qwen/qwen3.6-flash", "https://openrouter.ai/api/v1",    "CHEAP_API_KEY"),
    "qwen35b":   ("qwen/qwen3.6-35b-a3b", "https://openrouter.ai/api/v1",  "CHEAP_API_KEY"),
    "qwen-plus": ("qwen/qwen3.7-plus",  "https://openrouter.ai/api/v1",    "CHEAP_API_KEY"),
    "m3":        ("minimax/minimax-m3", "https://openrouter.ai/api/v1",    "CHEAP_API_KEY"),
    "gemini":    ("google/gemini-2.5-flash", "https://openrouter.ai/api/v1", "CHEAP_API_KEY"),
    "llama":     ("meta-llama/llama-4-maverick", "https://openrouter.ai/api/v1", "CHEAP_API_KEY"),
    "sonnet":    ("anthropic/claude-sonnet-4-20250514", "https://openrouter.ai/api/v1", "CHEAP_API_KEY"),
}
_tg_model: dict[int, str] = {}  # per-chat TG override
_WEB_MODEL = "flash"  # flaskchat executor model

# --- Persist model overrides ---
def _load_tg_models():
    """Load per-chat model overrides from DB into _tg_model."""
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("CREATE TABLE IF NOT EXISTS tg_model (chat_id INTEGER PRIMARY KEY, model TEXT)")
    for row in c.execute("SELECT chat_id, model FROM tg_model").fetchall():
        _tg_model[row[0]] = row[1]
    c.close()

def _save_tg_model(chat_id, model):
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("INSERT OR REPLACE INTO tg_model VALUES (?,?)", (chat_id, model))
    c.commit(); c.close()

def _clear_tg_model(chat_id):
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("DELETE FROM tg_model WHERE chat_id=?", (chat_id,)); c.commit(); c.close()

_load_tg_models()  # load on startup

# --- Transaction log (debugging) ---
def _log_transaction(platform: str, chat_id: str, model: str, query: str, debate: str, answer: str):
    """Log a full interaction (query → debate → answer) for debugging."""
    try:
        c = sqlite3.connect(str(TRANSACTIONS_DB))
        c.execute(
            "CREATE TABLE IF NOT EXISTS transactions "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, platform TEXT, "
            "chat_id TEXT, model TEXT, query TEXT, debate TEXT, answer TEXT)"
        )
        c.execute(
            "INSERT INTO transactions (ts, platform, chat_id, model, query, debate, answer) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), platform, str(chat_id), model, query, debate, answer)
        )
        # Prune to last TRANSACTION_LIMIT
        c.execute(
            "DELETE FROM transactions WHERE id NOT IN "
            "(SELECT id FROM transactions ORDER BY id DESC LIMIT ?)",
            (TRANSACTION_LIMIT,)
        )
        c.commit(); c.close()
    except Exception:
        pass  # never break the main flow for debug logging

# --- Tools ---
def _tool_web_search(ctx: RunContext[None], query: str) -> str:
    """Search the web for current info via local SearXNG (multi-engine)."""
    try:
        import urllib.request as _ur, urllib.parse
        url = f"http://127.0.0.1:8080/search?q={urllib.parse.quote(query)}&format=json&engines=google,duckduckgo"
        with _ur.urlopen(_ur.Request(url, headers={"User-Agent": "ThinkLLM/0.3"}), timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e: return f"Search unavailable: {e}"
    results = data.get("results", [])
    if not results: return "No results."
    return "\n\n".join(f"{i+1}. {r.get('title','?')}\n   {r.get('content','')}\n   {r.get('url','')}"
                       for i, r in enumerate(results[:8]))

def _tool_run_code(ctx: RunContext[None], code: str) -> str:
    """Execute Python code via n8n sandbox (has numpy, matplotlib, pandas)."""
    try:
        import urllib.request as _ur
        req = _ur.Request("http://localhost:8000/code/execute",
                          data=json.dumps({"code": code, "timeout": 15}).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer 987654321"}, method="POST")
        with _ur.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        if result.get("error"):
            return f"Error: {result['error']}"
        return str(result.get("stdout", result.get("output", result.get("result", "(no output)"))))[:4000]
    except Exception as e:
        return f"Code execution failed: {e}"

def _tool_generate_graph(ctx: RunContext[None], x: list, y: list, chart_type: str = "line",
                         title: str = "") -> str:
    """Generate a chart/graph. x: list of values (numbers or string labels).
    y: list of numeric values. chart_type: line, bar, pie, scatter, histogram."""
    try:
        import urllib.request as _ur
        payload = {"data": {"x": x, "y": y}, "graph_type": chart_type}
        if title: payload["data"]["title"] = title
        req = _ur.Request("http://localhost:8000/code/generate_graph",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer 987654321"},
                          method="POST")
        with _ur.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        if result.get("image"):
            # Decode base64 and upload to n8n for public URL
            import base64, tempfile
            b64_data = result["image"].split(",", 1)[-1] if "," in result["image"] else result["image"]
            img_data = base64.b64decode(b64_data)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_data)
                tmp_path = f.name
            # Upload via curl (multipart is simpler via subprocess)
            p = subprocess.run(
                ["curl", "-s", "http://localhost:8000/upload/file",
                 "-H", "Authorization: Bearer 987654321",
                 "-F", f"file=@{tmp_path}"],
                capture_output=True, text=True, timeout=20)
            Path(tmp_path).unlink(missing_ok=True)
            try:
                up_result = json.loads(p.stdout)
                if up_result.get("download_url"):
                    return f"Graph: {up_result['download_url']}"
            except json.JSONDecodeError:
                pass
            return f"Upload failed: {p.stdout[:200]}"
        return f"Graph generated but no image returned: {str(result)[:500]}"
    except Exception as e:
        return f"Graph generation failed: {e}"

def _tool_web_fetch(ctx: RunContext[None], url: str) -> str:
    """Fetch and extract text content from a web page URL."""
    try:
        import urllib.request as _ur
        req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0 ThinkLLM/0.3"})
        with _ur.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e: return f"Fetch failed: {e}"
    # Basic HTML-to-text: strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "No readable content."

def _tool_youtube_transcript(ctx: RunContext[None], url: str) -> str:
    """Get the transcript/subtitles of a YouTube video via VPS API."""
    try:
        import urllib.request as _ur
        req = _ur.Request(
            "http://141.11.17.227:8000/api/youtube/script",
            data=json.dumps({"video_url_or_id": url}).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": "987654321"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        if not data.get("success") or not data.get("transcript_available"):
            return f"No transcript available. Title: {data.get('metadata',{}).get('title','?')}"
        meta = data.get("metadata", {})
        lines = [f"Title: {meta.get('title','?')}",
                 f"Duration: {int(meta.get('duration',0))//60}min"]
        for seg in data["transcript"]:
            s = int(seg["start"])
            ts = f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
            lines.append(f"[{ts}] {seg['text']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Transcript unavailable: {e}"

def _tool_call_n8n(ctx: RunContext[None], workflow_name: str, payload: str = "{}") -> str:
    """Call an HTTP endpoint (n8n webhook or any API). Provide full URL or workflow name.
    Known n8n workflows: search-agent, yahoo-search, brave-search, bing-search,
    duckduck-search, rag-search, text-search, websearchapi.
    Base URL for n8n webhooks: https://n8n.smartdochub.net/webhook/"""
    import urllib.request as _ur

    n8n_map = {
        "search-agent": "https://n8n.smartdochub.net/webhook/search-agent",
        "yahoo-search": "https://n8n.smartdochub.net/webhook/yahoo-search",
        "brave-search": "https://n8n.smartdochub.net/webhook/brave-search",
        "bing-search": "https://n8n.smartdochub.net/webhook/bing-search",
        "duckduck-search": "https://n8n.smartdochub.net/webhook/duckduck-search",
        "rag-search": "https://n8n.smartdochub.net/webhook/rag-search",
        "text-search": "https://n8n.smartdochub.net/webhook/text-search",
        "websearchapi": "https://n8n.smartdochub.net/webhook/websearchapi",
    }
    # If it looks like a URL, use directly; otherwise look up in map
    url = workflow_name if workflow_name.startswith("http") else n8n_map.get(workflow_name.lower(), "")
    if not url:
        return f"Unknown: {workflow_name}. Available: {', '.join(n8n_map)} | or pass a direct URL"

    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        req = _ur.Request(url, data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=30) as r:
            result = r.read().decode("utf-8", errors="replace")
        return result[:4000] if result else "(empty response)"
    except Exception as e:
        return f"Call failed: {e}"

# --- App ---
app = FastAPI(title="ThinkLLM", version="0.3.0", docs_url=None, openapi_url=None)
_cache = DebateCache()
Path("/tmp/thinkllm_static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="/tmp/thinkllm_static"), name="static")

# Init TG history DB
def _init_db():
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("CREATE TABLE IF NOT EXISTS tg_history (chat_id INTEGER, query TEXT, answer TEXT, ts REAL)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tg ON tg_history(chat_id, ts)")
    c.commit(); c.close()
_init_db()


@app.post("/webhook/chat")
async def webhook_chat(request: Request):
    body = await request.json()
    # Shared secret check (FlaskChat sends Authorization header)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != _WEBHOOK_SECRET:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    query = body.get("chatinput", "").strip()
    chat_history = body.get("chat_history", "")
    skip_debate = request.query_params.get("skip_debate") == "1"
    if not query: return StreamingResponse(_err("Empty query"), media_type="text/plain")
    cfg = load_config(str(CONFIG_PATH))
    return StreamingResponse(_fc_stream(query, cfg, chat_history, skip_debate),
                             media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache"})


@app.post("/telegram/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if not TG_BOT_TOKEN or token != TG_BOT_TOKEN:
        print(f"[TG] rejected token={token[:10]}...", flush=True)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try: body = await request.json()
    except: return {"error": "bad json"}
    msg = body.get("message", {})
    cid = msg.get("chat", {}).get("id")
    text = (msg.get("text") or "").strip()
    if not cid or not text:
        print(f"[TG SKIP] cid={cid!r} text={text[:50]!r} keys={list(body.keys())} msg_keys={list(msg.keys())}", flush=True)
        return {"ok": True}
    clean = re.sub(r"(@\w+bot)$", "", text, flags=re.IGNORECASE).strip()

    if clean == "/start":
        await _tg_send(cid, "🤖 <b>ThinkLLM</b> — debate engine\n/cmds: /new /model\nSend any question.")
        return {"ok": True}
    if clean in ("/new", "/clear"):
        _cache.clear(); _clear_tg(cid)
        cur = _tg_model.get(cid, "flash")
        model_name = MODELS[cur][0]
        await _tg_send(cid, f"✅ New session. <b>{cur}</b> — {model_name}")
        return {"ok": True}
    if clean.startswith("/model"):
        await _handle_model(cid, clean); return {"ok": True}

    asyncio.create_task(_tg_query(cid, text))
    return {"ok": True}


# ============================================================
# FlaskChat
# ============================================================
async def _fc_stream(query, cfg, chat_history, skip_debate):
    error_msg = None
    try:
        yield _line("begin", metadata={"nodeName": "ThinkLLM"})
        ctx = f"Previous:\n{chat_history}\n\nNow: {query}" if chat_history else query
        if skip_debate:
            debate_text = "[No debate]"
        else:
            # Gatekeeper: classify + history pruning + web context
            _, primer, use_history = await _gatekeeper(query, chat_history)
            if not use_history:
                ctx = query  # drop unrelated history
            # Debaters: Flash (DeepSeek direct) ×2
            cfg.debater_a.model = "deepseek-v4-flash"
            cfg.debater_a.base_url = "https://api.deepseek.com/v1"
            cfg.debater_a.api_key = os.environ.get("EXPENSIVE_API_KEY")
            cfg.debater_b.model = "deepseek-v4-flash"
            cfg.debater_b.base_url = "https://api.deepseek.com/v1"
            cfg.debater_b.api_key = os.environ.get("EXPENSIVE_API_KEY")
            engine = ThinkLLM(cfg, cache=_cache)
            tx = await _debate(engine, ctx, primer=primer)
            debate_text = _debate_transcript(tx, name_a="Critical Analyst", name_b="Constructive Builder")

        # Web uses Qwen for proper markdown formatting
        web_mid, web_burl, web_kenv = MODELS.get(_WEB_MODEL, MODELS["qwen-plus"])
        web_key = os.environ.get(web_kenv, os.environ["OPENAI_API_KEY"])
        agent = _make_agent(web_mid, web_burl, web_key, platform="flaskchat")
        user_msg = (f"USER QUERY: {ctx}\n\nDEBATE TRANSCRIPT:\n{debate_text}\n\n"
                    f"Produce the final answer. Use tools if needed.")
        result = await agent.run(user_msg)
        # Post-process markdown for web rendering
        text = _polish_markdown(result.output)
        _log_transaction("web", "flaskchat", web_mid, query, debate_text, result.output)
        for i in range(0, len(text), 2000):
            yield _line("item", content=text[i:i+2000])
    except Exception as e:
        error_msg = str(e)[:200]
        yield _line("item", content=f"\n\nError: {error_msg}")
    finally:
        yield _line("end", metadata={"nodeName": "ThinkLLM", "error": error_msg})


# ============================================================
# Telegram
# ============================================================
async def _tg_query(chat_id: int, query: str):
    import httpx
    stop = asyncio.Event()

    async def _typing():
        while not stop.is_set():
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    await c.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendChatAction",
                                 json={"chat_id": chat_id, "action": "typing"})
            except: pass
            try: await asyncio.wait_for(stop.wait(), 4)
            except asyncio.TimeoutError: pass

    asyncio.create_task(_typing())
    sent_id = None

    try:
        cfg = load_config(str(CONFIG_PATH))
        history = _load_tg(chat_id)
        ctx = _build_ctx(history)

        platform = ("[You are on Telegram. Use <b>bold</b>, <i>italic</i>, <code>code</code>, "
                    "<pre>blocks</pre>, <a href>links</a>. No tables — use lists. Under 3500 chars.]\n\n")
        full_query = platform + query

        # Resolve model
        key = _tg_model.get(chat_id, "flash")
        mid, burl, kenv = MODELS.get(key, MODELS["flash"])
        api_key = os.environ.get(kenv, os.environ["OPENAI_API_KEY"])
        print(f"[TG] chat={chat_id} model={key} ({mid})", flush=True)

        # Debaters: Flash ×2
        cfg.debater_a.model = "deepseek-v4-flash"
        cfg.debater_a.base_url = "https://api.deepseek.com/v1"
        cfg.debater_a.api_key = os.environ.get("EXPENSIVE_API_KEY")
        cfg.debater_b.model = "deepseek-v4-flash"
        cfg.debater_b.base_url = "https://api.deepseek.com/v1"
        cfg.debater_b.api_key = os.environ.get("EXPENSIVE_API_KEY")

        # Gatekeeper: classify + history pruning + web context
        _, primer, use_history = await _gatekeeper(query, ctx)
        if use_history and ctx:
            full_query = f"{platform}Previous:\n{ctx}\n\nNow: {query}"

        engine = ThinkLLM(cfg, cache=_cache)
        tx = await _debate(engine, full_query, primer=primer, cache_query=query)
        debate_text = _debate_transcript(tx)

        # Executor via pydantic-ai agent
        agent = _make_agent(mid, burl, api_key)
        user_msg = (f"USER QUERY: {full_query}\n\nDEBATE TRANSCRIPT:\n{debate_text}\n\n"
                    f"Produce the final answer. Use tools if needed.")

        # Run agent (handles tool calls), then stream output
        result = await agent.run(user_msg)
        final_text = result.output

        # Stream the output in chunks for Telegram
        accumulated = ""
        last_edit = 0
        words = final_text.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            accumulated += chunk
            now = time.time()
            if now - last_edit > 1.5 or i == len(words) - 1:
                preview = accumulated[:TG_CHAR_LIMIT - 5] + (" ✍️" if i < len(words) - 1 else "")
                if sent_id is None:
                    sent_id = await _tg_send(chat_id, preview, return_id=True, parse_mode=None)
                else:
                    await _tg_edit(chat_id, sent_id, preview, parse_mode=None)
                last_edit = now

        stop.set()
        answer_html = _md_to_tg(final_text) + f"\n\n<i>🤖 {mid}</i>"
        # Detect image URLs for inline link preview
        img_urls = re.findall(r'https?://\S+\.(?:png|jpg|jpeg|gif|webp)(?:\?\S*)?', final_text)
        preview_url = img_urls[0] if img_urls else None
        if sent_id is not None:
            await _tg_edit(chat_id, sent_id, answer_html, link_preview_url=preview_url)
        else:
            await _tg_send(chat_id, answer_html, link_preview_url=preview_url)
        _save_tg(chat_id, query, accumulated)
        _log_transaction("telegram", str(chat_id), mid, query, debate_text, final_text)

    except Exception as e:
        stop.set()
        import traceback; traceback.print_exc()
        await _tg_send(chat_id, f"❌ {str(e)[:200]}")


# ============================================================
# Markdown post-processing
# ============================================================
def _polish_markdown(text: str) -> str:
    """Post-process LLM markdown for web rendering — normalize spacing, fix quirks."""
    # Fix inline headings: "text ## Heading" → "text\n\n## Heading"
    text = re.sub(r"([^\n#])(#{1,4}\s+\w)", r"\1\n\n\2", text)
    # Ensure blank line before headings
    text = re.sub(r"([^\n])\n(#{1,4} )", r"\1\n\n\2", text)
    # Ensure blank line before code blocks
    text = re.sub(r"([^\n])\n(```)", r"\1\n\n\2", text)
    # Ensure blank line before lists
    text = re.sub(r"([^\n])\n(- |\d+\. )", r"\1\n\n\2", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# Agent factory
# ============================================================
def _make_agent(model_id: str, base_url: str, api_key: str, platform: str = "telegram") -> Agent:
    """Create a Pydantic AI agent configured for a specific platform."""
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model = OpenAIChatModel(
        model_id, provider=provider,
        settings=ModelSettings(thinking="xhigh"),
    )

    plan_rules = (
        "PLAN ENFORCEMENT (CRITICAL — FOLLOW EXACTLY):\n"
        "- The debate transcript below ends with a PLAN: step1 → step2 → step3.\n"
        "- You MUST execute EVERY step in order. Never skip, never reorder.\n"
        "- If PLAN calls run_code: call run_code FIRST, then show the output.\n"
        "- If PLAN specifies an output format (poem, table, specific structure):\n"
        "  produce that EXACT format, but ALSO include your work and verification.\n"
        "- If PLAN calls for validation (checking for 'e' in text, verifying primes, etc.):\n"
        "  DO the validation. Show the validation results.\n"
        "- NEVER output only the final artifact — always show calculation/verification steps.\n"
        "- The debate PLAN is the agreed strategy. You are the executor. Execute it.\n"
    )
    search_rules = (
        "SEARCH METHOD (CRITICAL):\n"
        "- For fact-checking ('does X exist?', 'when did Y happen?'): "
        "start with a broad web_search first. Wikipedia is your most reliable single source.\n"
        "- If first search returns nothing, ESCALATE — try different phrasing, "
        "broader terms, or Wikipedia directly. Do NOT conclude 'doesn't exist.'\n"
        "- 'No results in source X' ≠ 'doesn't exist.' "
        "Say 'I couldn't find it in [X]' — never claim certainty from negative results.\n"
        "- When a user corrects you and you verify their correction: ACCEPT it. "
        "The user being right is not sycophancy. Do not re-litigate verified facts.\n"
    )
    math_rules = (
        "MATH VERIFICATION (CRITICAL):\n"
        "- The debate transcript below may contain NUMERICAL ERRORS. "
        "Debaters have NO tools and compute everything in their head — they get math wrong.\n"
        "- NEVER trust pre-computed numbers from the debate. "
        "For any calculation (salary, budget, loan, percentages, unit conversions, arithmetic): "
        "call run_code to recompute from first principles.\n"
        "- Write the full calculation as a Python script, print results clearly, run it. "
        "This applies even if the numbers \"look right\" — the debate's RM 7,375 - RM 4,450 "
        "was claimed as RM 725 when the real answer is RM 2,925.\n"
        "- If the debate's numbers disagree with run_code output: use run_code's output. "
        "The debate is advisory strategy only — its math is UNTRUSTED.\n"
    )

    if platform == "telegram":
        system = (
            f"Today is {time.strftime('%A, %d %B %Y')}. "
            "You are on Telegram. Formatting rules:\n"
            "• <b>bold</b> for key terms and headings\n"
            "• <i>italic</i> for emphasis, titles, source names\n"
            "• <code>monospace</code> for dates, numbers, genres\n"
            "• <a href=\"url\">link text</a> for sources\n"
            "• Use • bullet lists (not markdown - or *)\n"
            "• Separate sections with blank lines (use — as visual break if needed)\n"
            "• Every supported tag: <b> <i> <code> <pre> <a> — nothing else renders\n"
            "No markdown tables — use lists. "
            "Keep responses under 3500 chars. Use web_search for any recent info. "
            "Graphs: generate_graph returns a URL — share the link with a brief description. "
            "IMPORTANT: When asked to run/compute/calculate, call the tools — never skip execution. "
            "CRITICAL — DEBATE MAY LIE ABOUT YOUR CAPABILITIES: "
            "The debate transcript is strategy ONLY. Debaters have NO tools and often "
            "falsely claim 'we lack live access' or 'we cannot search the web.' "
            "You HAVE web_search, web_fetch, youtube_transcript — use them. "
            "If the debate says 'redirect user to external sources' but you have "
            "a tool that can answer directly: IGNORE the debate and call the tool. "
            "You ALWAYS have internet access via your tools. Never claim otherwise. "
            "Never mention internal debate process.\n"
            + plan_rules + search_rules + math_rules
        )
    else:  # flaskchat / web
        system = (
            f"Today is {time.strftime('%A, %d %B %Y')}. "
            "You are on a web chat interface with full markdown rendering. "
            "Use rich formatting: ## headings, **bold**, *italic*, `code`, "
            "```blocks```, | tables |, - lists, > blockquotes. "
            "CRITICAL: Tables MUST be multi-line. "
            "Graphs: use ![description](url) to display generate_graph results inline. "
            "IMPORTANT: When asked to run/compute/calculate/execute code, "
            "you MUST call run_code or generate_graph tools — "
            "never just show code without executing it. "
            "You have unlimited space — be thorough and detailed. "
            "Use web_search for any recent or current information. "
            "Never mention internal debate process.\n"
            + plan_rules + search_rules + math_rules
        )

    return Agent(
        model,
        tools=[
            Tool(_tool_web_search), Tool(_tool_run_code),
            Tool(_tool_generate_graph),
            Tool(_tool_web_fetch), Tool(_tool_youtube_transcript),
            Tool(_tool_call_n8n),
        ],
        system_prompt=system,
    )


# ============================================================
# Gatekeeper — classify + expand context for ALL query types
# ============================================================
_GATEKEEPER_MODEL = "deepseek-v4-flash"
_GATEKEEPER_URL = "https://api.deepseek.com/v1"

async def _gatekeeper(query: str, chat_history: str = ""):
    """Classify query, check history relevance, and expand context.
    Returns (classification, primer, use_history).
    URL → youtube_transcript or web_fetch context
    FACT/COMPLEX → web_search context
    On failure → ("COMPLEX", None, True), debate runs without context.
    """
    import httpx, urllib.request as _ur
    key = os.environ.get("EXPENSIVE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not key:
        return ("COMPLEX", None, True)

    # --- Step 0: History pruning ---
    use_history = False
    if chat_history:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{_GATEKEEPER_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": _GATEKEEPER_MODEL,
                        "messages": [{"role": "user", "content": (
                            f"Previous conversation (last few exchanges):\n{chat_history[:400]}\n\n"
                            f"New query: \"{query[:200]}\"\n\n"
                            f"Is the new query a follow-up to or about the same topic as the previous? "
                            f"Answer one word: RELATED or UNRELATED"
                        )}],
                        "max_tokens": 5,
                        "temperature": 0,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                result = resp.json()["choices"][0]["message"]["content"].strip().upper()
                use_history = "RELATED" in result
        except Exception:
            use_history = bool(chat_history)  # default to keeping history on error

    # --- Step 1: Classify ---
    # URL detection is regex-first — don't trust LLM for this
    if re.search(r'https?://', query):
        kind = "URL"
    else:
        prompt = (
            f'Classify: "{query[:300]}"\\n'
            f"One word: FACT | COMPLEX\\n"
            f"FACT=simple lookup/calc, COMPLEX=needs reasoning"
        )
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{_GATEKEEPER_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": _GATEKEEPER_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "temperature": 0,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                classification = resp.json()["choices"][0]["message"]["content"].strip().upper()
                kind = "FACT" if "FACT" in classification else "COMPLEX"
        except Exception:
            kind = "COMPLEX"

    # --- Step 2: Expand context ---
    context = None
    try:
        if kind == "URL":
            url_match = re.search(r'https?://\S+', query)
            if url_match:
                url = url_match.group(0).rstrip(")&")
                if "youtube.com" in url or "youtu.be" in url:
                    req = _ur.Request(
                        "http://141.11.17.227:8000/api/youtube/script",
                        data=json.dumps({"video_url_or_id": url}).encode(),
                        headers={"Content-Type": "application/json", "X-API-Key": "987654321"},
                        method="POST",
                    )
                    with _ur.urlopen(req, timeout=15) as r:
                        data = json.loads(r.read())
                    if data.get("success") and data.get("transcript_available"):
                        meta = data.get("metadata", {})
                        title = meta.get("title", "Unknown")
                        dur = int(meta.get("duration", 0)) // 60
                        text = " ".join(s["text"] for s in data["transcript"][:12])
                        context = f"VIDEO: \\\"{title}\\\" (~{dur}min). Preview: {text[:500]}"
                    elif data.get("metadata", {}).get("title"):
                        context = f"VIDEO: \\\"{data['metadata']['title']}\\\" (no transcript)"
                else:
                    req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0 ThinkLLM/0.3"})
                    with _ur.urlopen(req, timeout=10) as r:
                        html = r.read().decode("utf-8", errors="replace")
                    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL|re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL|re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\\s+", " ", text).strip()[:500]
                    title_m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
                    title = title_m.group(1).strip() if title_m else "Web page"
                    context = f"WEB: \\\"{title}\\\". Preview: {text[:400]}"
        else:
            # FACT or COMPLEX — web search for ground truth
            sq = query[:200]
            import urllib.parse as _uparse
            sr = _ur.Request(
                f"http://127.0.0.1:8080/search?q={_uparse.quote(sq)}&format=json&engines=google,duckduckgo",
                headers={"User-Agent": "ThinkLLM/0.3"},
            )
            with _ur.urlopen(sr, timeout=8) as r:
                results = json.loads(r.read()).get("results", [])
            if results:
                snippets = []
                for i, res in enumerate(results[:3]):
                    snippets.append(f"{i+1}. {res.get('title','?')}: {res.get('content','')[:200]}")
                context = "WEB SEARCH:\\n" + "\\n".join(snippets)
    except Exception:
        context = None

    # --- Step 3: Generate opening debate statement ---
    primer = None
    if context:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{_GATEKEEPER_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": _GATEKEEPER_MODEL,
                        "messages": [{"role": "user", "content": (
                            f"You are a gatekeeper AI. You fetched this content for a strategy debate:\n\n"
                            f"{context[:800]}\n\n"
                            f"Write the OPENING STATEMENT for the debate. The two debaters will respond to you.\n"
                            f"Suggest: how the Executor should structure the answer, what to emphasize,\n"
                            f"what pitfalls to avoid, which tools to use. Be concise — under 500 chars.\n"
                            f"Do NOT write the final answer. Write strategy recommendations.\n"
                            f"Start with: 'I've reviewed the content. Here is my strategy recommendation:'"
                        )}],
                        "max_tokens": 300,
                        "temperature": 0.3,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                primer = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            primer = context  # fallback: use raw context as primer

    print(f"[GATEKEEPER] kind={kind} primer={'YES' if primer else 'NONE'} history={'KEEP' if use_history else 'DROP'}", flush=True)
    return (kind, primer, use_history)


# ============================================================
# Debate
# ============================================================
def _extract_text(msg) -> str:
    """Extract text content from a pydantic-ai ModelMessage."""
    parts = getattr(msg, "parts", [])
    return " ".join(getattr(p, "content", "") or "" for p in parts)

def _debate_transcript(tx, name_a="Critical Analyst", name_b="Constructive Builder"):
    """Convert debate transcript to readable text. Handles gatekeeper primer if present."""
    _cl = re.compile(r"<invoke\\b[^>]*>.*?</invoke>", re.DOTALL)
    lines = []
    has_primer = len(tx) > 1 and len(tx) % 2 == 0  # even total = primer present
    print(f"[TRANSCRIPT] len={len(tx)} has_primer={has_primer}", flush=True)
    start = 1
    if has_primer:
        lines.append(f"[Gatekeeper]: {_cl.sub('[tool removed]', _extract_text(tx[1]))}")
        start = 2
    for i, m in enumerate(tx[start:], start=start):
        # Without primer: odd idx = debater A. With primer: even offset — adjust
        idx = i - (1 if has_primer else 0)
        name = name_a if idx % 2 == 1 else name_b
        lines.append(f"[{name}]: {_cl.sub('[tool removed]', _extract_text(m))}")
    return "\n\n".join(lines)

_DEBATE_TOOLS_MSG = """\
I am an AI that needs to answer this user question:

"%s"

I have access to these tools:
• web_search(query) — live internet search (Google + DuckDuckGo)
• web_fetch(url) — extract full text from any webpage
• youtube_transcript(url) — get video subtitles with timestamps
• run_code(code) — Python sandbox (numpy, pandas, matplotlib)
• generate_graph(x, y, type, title) — create charts (returns image URL)
• call_n8n(name, payload) — external API/webhook calls

Debate the BEST strategy. I CAN use web_search for live data — NOT limited to training.
Do NOT suggest redirecting users — I have tools to answer directly.

FORMAT: This is machine-to-machine. NO markdown, NO prose, NO headings.
Use compressed shorthand: fragments, symbols, arrows (→), minimal tokens.
End with: PLAN: step1 → step2 → step3."""

async def _debate(engine, query, primer: str = None, cache_query: str = None):
    """Run debate. If primer is provided, it's inserted as the first agent message
    (gatekeeper opening statement) and debaters respond to it.
    
    query: full context for debate (may include chat history)
    cache_query: raw user query for cache key (avoids history poisoning)"""
    debate_query = _DEBATE_TOOLS_MSG % query
    tx = [ModelRequest(parts=[UserPromptPart(content=debate_query)], timestamp=datetime.now(timezone.utc))]
    if primer:
        tx.append(ModelResponse(parts=[TextPart(content=primer)], timestamp=datetime.now(timezone.utc)))
        print(f"[DEBATE] Primer injected: {len(primer)} chars", flush=True)
    # Use cache_query for lookup, fall back to query if not provided
    lookup = cache_query or query
    cached = engine._load_cache(lookup)
    if cached: return cached
    for turn in range(engine.config.max_turns):
        ra = await engine.agent_a.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=ra)], timestamp=datetime.now(timezone.utc)))
        rb = await engine.agent_b.respond(tx) or ""
        tx.append(ModelResponse(parts=[TextPart(content=rb)], timestamp=datetime.now(timezone.utc)))
        if engine.config.early_termination and _has_converged(ra, rb): break
    engine._save_cache(lookup, tx)
    print(f"[DEBATE] tx length={len(tx)} primer={'YES' if primer else 'NO'}", flush=True)
    return tx


# ============================================================
# Model command
# ============================================================
async def _handle_model(chat_id, text):
    parts = text.split(" ", 1)
    if len(parts) == 1:
        cur = _tg_model.get(chat_id, "flash")
        lst = "\n".join(f"• <code>/model {k}</code> — {v[0]}" for k, v in MODELS.items())
        await _tg_send(chat_id, f"<b>Current:</b> {cur}\n\n<b>Available:</b>\n{lst}\n\n<code>/model default</code>")
        return
    m = parts[1].strip().lower()
    if m in MODELS:
        _tg_model[chat_id] = m
        _save_tg_model(chat_id, m)
        await _tg_send(chat_id, f"✅ <b>{m}</b> ({MODELS[m][0]})")
    elif m == "default":
        _tg_model.pop(chat_id, None)
        _clear_tg_model(chat_id)
        await _tg_send(chat_id, "✅ Reset to default.")
    else:
        await _tg_send(chat_id, f"Unknown. Use: {', '.join(MODELS)} | default")


# ============================================================
# TG history (SQLite)
# ============================================================
def _load_tg(chat_id):
    c = sqlite3.connect(str(HISTORY_DB))
    rows = c.execute("SELECT query, answer FROM tg_history WHERE chat_id=? ORDER BY ts", (chat_id,)).fetchall()
    c.close(); return [(r[0], r[1]) for r in rows]

def _save_tg(chat_id, q, a):
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("INSERT INTO tg_history VALUES (?,?,?,?)", (chat_id, q, a, time.time()))
    c.commit(); c.close()

def _clear_tg(chat_id):
    c = sqlite3.connect(str(HISTORY_DB))
    c.execute("DELETE FROM tg_history WHERE chat_id=?", (chat_id,)); c.commit(); c.close()

def _build_ctx(history):
    if not history: return ""
    total = len(history)
    recent_chars = 0; recent_idx = 0
    for i in range(total - 1, -1, -1):
        q, a = history[i]
        if recent_chars + len(q) + min(len(a), 300) + 30 > 6000:
            recent_idx = i + 1; break
        recent_chars += len(q) + min(len(a), 300) + 30
    else: recent_idx = 0
    lines = []
    if recent_idx > 0:
        old = ", ".join(q for q, _ in history[:recent_idx][:10])
        if len(history[:recent_idx]) > 10: old += f" (+{len(history[:recent_idx])-10})"
        lines.append(f"[Earlier: {old}]")
    for q, a in history[recent_idx:]:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a[:300]}")
    return "\n".join(lines)


# ============================================================
# TG messaging
# ============================================================
async def _tg_send(chat_id, text, return_id=False, parse_mode="HTML", link_preview_url=None):
    import httpx
    if parse_mode == "HTML": text = _sanitize(text)
    payload = {"chat_id": chat_id, "text": text[:TG_CHAR_LIMIT]}
    if link_preview_url:
        payload["link_preview_options"] = {"url": link_preview_url, "prefer_large_media": True}
    else:
        payload["link_preview_options"] = {"is_disabled": True}
    if parse_mode: payload["parse_mode"] = parse_mode
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", json=payload)
            if r.status_code == 200 and return_id: return r.json()["result"]["message_id"]
            if r.status_code != 200: print(f"[TG SEND {r.status_code}] {r.text[:200]}", flush=True)
    except Exception as e: print(f"[TG SEND FAIL] {e}", flush=True)

async def _tg_edit(chat_id, msg_id, text, parse_mode="HTML", link_preview_url=None):
    import httpx
    if parse_mode == "HTML": text = _sanitize(text)
    payload = {"chat_id": chat_id, "message_id": msg_id, "text": text[:TG_CHAR_LIMIT]}
    if link_preview_url:
        payload["link_preview_options"] = {"url": link_preview_url, "prefer_large_media": True}
    else:
        payload["link_preview_options"] = {"is_disabled": True}
    if parse_mode: payload["parse_mode"] = parse_mode
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/editMessageText", json=payload)
            if r.status_code != 200: print(f"[TG EDIT {r.status_code}] {r.text[:150]}", flush=True)
    except Exception as e: print(f"[TG EDIT FAIL] {e}", flush=True)


def _sanitize(text):
    prot = []
    def _p(m): prot.append(m.group(0)); return f"%%P{len(prot)-1}%%"
    text = re.sub(r"</?(?:b|i|u|code|pre|a)(?:\s[^>]*)?>|&\w+;", _p, text)
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    for i, p in enumerate(prot): text = text.replace(f"%%P{i}%%", p)
    return text

def _md_to_tg(text):
    cbs = []
    def _scb(m): cbs.append(m.group(2).strip()); return f"%%CB{len(cbs)-1}%%"
    text = re.sub(r"```(?:\w+)?\n(.*?)```", _scb, text, flags=re.DOTALL)
    ics = []
    def _sic(m): ics.append(m.group(1)); return f"%%IC{len(ics)-1}%%"
    text = re.sub(r"`([^`]+)`", _sic, text)
    for pat, rep in [(r"^#### (.+)$", r"<b><u>\1</u></b>"), (r"^### (.+)$", r"<b>\1</b>"),
                     (r"^## (.+)$", r"<b>\1</b>"), (r"^# (.+)$", r"<b>\1</b>")]:
        text = re.sub(pat, rep, text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"^---+$", "————————", text, flags=re.MULTILINE)
    text = re.sub(r"^> (.+)$", r"<i>▎ \1</i>", text, flags=re.MULTILINE)
    for i, cb in enumerate(cbs): text = text.replace(f"%%CB{i}%%", f"<pre>{cb}</pre>")
    for i, ic in enumerate(ics): text = text.replace(f"%%IC{i}%%", f"<code>{ic}</code>")
    return text


# ============================================================
# Utils
# ============================================================
async def _err(msg):
    yield _line("begin", metadata={"nodeName": "ThinkLLM"})
    yield _line("item", content=f"Error: {msg}")
    yield _line("end", metadata={"error": msg})

def _line(t, **kw):
    p = {"type": t}
    if "content" in kw: p["content"] = kw["content"]
    p["metadata"] = kw.get("metadata", {"timestamp": int(time.time() * 1000)})
    return json.dumps(p, ensure_ascii=False) + "\n"


# ============================================================
# OpenAI-compatible API (for OpenWebUI)
# ============================================================
_OWUI_API_KEY = os.environ.get("OPENWEBUI_API_KEY", "987654321")

def _check_owui_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return token == _OWUI_API_KEY
    return False

@app.get("/v1/models")
async def openai_models(request: Request):
    """Return available models in OpenAI format for OpenWebUI discovery."""
    if not _check_owui_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return {
        "object": "list",
        "data": [
            {"id": "thinkllm-debate", "object": "model", "owned_by": "thinkllm"},
            {"id": "thinkllm-fast", "object": "model", "owned_by": "thinkllm"},
        ]
    }

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    """OpenAI-compatible chat completions with debate pipeline."""
    if not _check_owui_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", True)

    # Extract last user message + build history from prior messages
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return StreamingResponse(_sse_err("No user message"), media_type="text/event-stream")

    query = user_msgs[-1].get("content", "").strip()
    # Build chat history from all messages except the last user one
    history_parts = []
    for m in messages:
        if m is user_msgs[-1]:
            break
        history_parts.append(f"[{m.get('role','?')}]: {m.get('content','')[:500]}")
    chat_history = "\n".join(history_parts[-20:])  # last 20 messages max

    if not stream:
        return await _openai_sync(query, chat_history)

    return StreamingResponse(
        _openai_stream(query, chat_history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

async def _openai_stream(query: str, chat_history: str):
    """Run debate pipeline and yield SSE chunks."""
    try:
        cfg = load_config(str(CONFIG_PATH))
        # Debaters: Flash ×2
        cfg.debater_a.model = "deepseek-v4-flash"
        cfg.debater_a.base_url = "https://api.deepseek.com/v1"
        cfg.debater_a.api_key = os.environ.get("EXPENSIVE_API_KEY")
        cfg.debater_b.model = "deepseek-v4-flash"
        cfg.debater_b.base_url = "https://api.deepseek.com/v1"
        cfg.debater_b.api_key = os.environ.get("EXPENSIVE_API_KEY")

        ctx = f"Previous:\n{chat_history}\n\nNow: {query}" if chat_history else query

        # Gatekeeper: classify + history pruning + web context
        _, primer, use_history = await _gatekeeper(query, chat_history)
        if not use_history:
            ctx = query  # drop unrelated history

        engine = ThinkLLM(cfg, cache=_cache)
        tx = await _debate(engine, ctx, primer=primer)
        debate_text = _debate_transcript(tx)

        # Executor: qwen-plus (good markdown for web)
        web_mid, web_burl, web_kenv = MODELS.get(_WEB_MODEL, MODELS["qwen-plus"])
        web_key = os.environ.get(web_kenv, os.environ["OPENAI_API_KEY"])
        agent = _make_agent(web_mid, web_burl, web_key, platform="flaskchat")
        user_msg = (f"USER QUERY: {ctx}\n\nDEBATE TRANSCRIPT:\n{debate_text}\n\n"
                    f"Produce the final answer. Use tools if needed.")
        result = await agent.run(user_msg)
        text = _polish_markdown(result.output)
        _log_transaction("openwebui", "openwebui", web_mid, query, debate_text, result.output)
        yield _sse_chunk(text)
    except Exception as e:
        yield _sse_chunk(f"\n\nError: {str(e)[:200]}")
    finally:
        yield "data: [DONE]\n\n"

async def _openai_sync(query: str, chat_history: str):
    """Non-streaming OpenAI response."""
    # Same as stream but collect all output
    cfg = load_config(str(CONFIG_PATH))
    cfg.debater_a.model = "deepseek-v4-flash"
    cfg.debater_a.base_url = "https://api.deepseek.com/v1"
    cfg.debater_a.api_key = os.environ.get("EXPENSIVE_API_KEY")
    cfg.debater_b.model = "deepseek-v4-flash"
    cfg.debater_b.base_url = "https://api.deepseek.com/v1"
    cfg.debater_b.api_key = os.environ.get("EXPENSIVE_API_KEY")

    ctx = f"Previous:\n{chat_history}\n\nNow: {query}" if chat_history else query

    # Gatekeeper: classify + history pruning + web context
    _, primer, use_history = await _gatekeeper(query, chat_history)
    if not use_history:
        ctx = query  # drop unrelated history

    engine = ThinkLLM(cfg, cache=_cache)
    tx = await _debate(engine, ctx, primer=primer)
    debate_text = _debate_transcript(tx)

    web_mid, web_burl, web_kenv = MODELS.get(_WEB_MODEL, MODELS["qwen-plus"])
    web_key = os.environ.get(web_kenv, os.environ["OPENAI_API_KEY"])
    agent = _make_agent(web_mid, web_burl, web_key, platform="flaskchat")
    user_msg = (f"USER QUERY: {ctx}\n\nDEBATE TRANSCRIPT:\n{debate_text}\n\n"
                f"Produce the final answer. Use tools if needed.")
    result = await agent.run(user_msg)
    text = _polish_markdown(result.output)
    _log_transaction("openwebui", "openwebui", web_mid, query, debate_text, result.output)

    return {
        "id": f"thinkllm-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "thinkllm-debate",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }

def _sse_chunk(content: str) -> str:
    return f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"

async def _sse_err(msg: str):
    yield _sse_chunk(f"Error: {msg}")
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    print(f"ThinkLLM v0.3 (pydantic-ai) — http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"Tools: web_search, run_code | Models: {len(MODELS)}")
    uvicorn.run("server:app", host=HTTP_HOST, port=HTTP_PORT, log_level="info")
