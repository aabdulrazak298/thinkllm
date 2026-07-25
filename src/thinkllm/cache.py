from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .types import DebateConfig, Message


class DebateCache:
    def __init__(self, db_path: str | Path = ""):
        if not db_path:
            db_path = Path.home() / ".thinkllm" / "cache.db"
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS debates (
                query_hash TEXT,
                config_hash TEXT,
                query TEXT NOT NULL,
                max_turns INTEGER NOT NULL,
                transcript_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (query_hash, config_hash)
            )"""
        )
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    @staticmethod
    def _config_fingerprint(config: DebateConfig) -> str:
        parts = [
            config.debater_a.name,
            config.debater_a.model,
            config.debater_a.provider,
            config.debater_a.system_prompt,
            config.debater_b.name,
            config.debater_b.model,
            config.debater_b.provider,
            config.debater_b.system_prompt,
            str(config.max_turns),
        ]
        return DebateCache._hash("|".join(parts))

    def get(self, query: str, config: DebateConfig) -> tuple[list[Message], int] | None:
        qh = self._hash(query)
        ch = self._config_fingerprint(config)
        row = self._conn.execute(
            "SELECT transcript_json, max_turns FROM debates WHERE query_hash=? AND config_hash=?",
            (qh, ch),
        ).fetchone()
        if row is None:
            return None
        messages_raw = json.loads(row[0])
        messages = [Message(**m) for m in messages_raw]
        return messages, row[1]

    def set(self, query: str, config: DebateConfig, transcript: list[Message]) -> None:
        qh = self._hash(query)
        ch = self._config_fingerprint(config)
        transcript_json = json.dumps([{"role": m.role, "content": m.content, "name": m.name} for m in transcript])
        self._conn.execute(
            "INSERT OR REPLACE INTO debates (query_hash, config_hash, query, max_turns, transcript_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (qh, ch, query, config.max_turns, transcript_json, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
