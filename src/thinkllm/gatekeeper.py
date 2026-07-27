"""Gatekeeper — classifies queries, prunes chat history, fetches web context."""

from __future__ import annotations

import json
import os
import re


async def run_gatekeeper(
    query: str,
    chat_history: str = "",
    *,
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com/v1",
    api_key: str | None = None,
    verbose: bool = False,
) -> tuple[str, str | None, bool]:
    """Classify query, check history relevance, and expand context.

    Returns (classification, primer, use_history).
    classification: "URL" | "FACT" | "COMPLEX"
    primer: debate opening statement with web context, or None on failure
    use_history: whether to include chat history in debate context
    """
    import httpx
    import urllib.request as _ur

    key = api_key or os.environ.get("EXPENSIVE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not key:
        return ("COMPLEX", None, True)

    # ── Step 0: History pruning ──
    use_history = True  # default: keep history
    if chat_history:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": (
                            f"Previous conversation (last few exchanges):\n{chat_history[:400]}\n\n"
                            f'New query: "{query[:200]}"\n\n'
                            f"Is the new query a follow-up to or about the same topic as the previous? "
                            f"Answer one word: RELATED or UNRELATED"
                        )}],
                        "max_tokens": 5,
                        "temperature": 0,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                result = resp.json()["choices"][0]["message"]["content"].strip().upper()
                use_history = "RELATED" in result or "UNRELATED" not in result
        except Exception:
            pass  # keep default True on error

    # ── Step 1: Classify ──
    if re.search(r"https?://", query):
        kind = "URL"
    else:
        prompt = (
            f'Classify: "{query[:300]}"\n'
            f"One word: FACT | COMPLEX\n"
            f"FACT=simple lookup/calc, COMPLEX=needs reasoning"
        )
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
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

    # ── Step 2: Expand context ──
    context = None
    try:
        if kind == "URL":
            url_match = re.search(r"https?://\S+", query)
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
                        context = f'VIDEO: "{title}" (~{dur}min). Preview: {text[:500]}'
                    elif data.get("metadata", {}).get("title"):
                        context = f'VIDEO: "{data["metadata"]["title"]}" (no transcript)'
                else:
                    req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0 ThinkLLM/0.3"})
                    with _ur.urlopen(req, timeout=10) as r:
                        html = r.read().decode("utf-8", errors="replace")
                    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()[:500]
                    title_m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
                    title = title_m.group(1).strip() if title_m else "Web page"
                    context = f'WEB: "{title}". Preview: {text[:400]}'
        else:
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
                    snippets.append(f"{i + 1}. {res.get('title', '?')}: {res.get('content', '')[:200]}")
                context = "WEB SEARCH:\n" + "\n".join(snippets)
    except Exception:
        context = None

    # ── Step 3: Generate opening debate statement ──
    primer = None
    if context:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
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
            primer = context

    if verbose:
        print(
            f"[GATEKEEPER] kind={kind} primer={'YES' if primer else 'NONE'} "
            f"history={'KEEP' if use_history else 'DROP'}",
            flush=True,
        )
    return (kind, primer, use_history)
