import hashlib
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

from .types import DebateConfig


class DebateCache:
    def __init__(self, db_path: str | Path = "", memory_cache_size: int = 1000):
        if not db_path:
            db_path = Path.home() / ".thinkllm" / "cache.db"
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")
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

        self._memory: OrderedDict[str, tuple[list, int]] = OrderedDict()
        self._memory_max = memory_cache_size

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
            str(config.debater_a.temperature),
            str(config.debater_a.top_p),
            config.debater_b.name,
            config.debater_b.model,
            config.debater_b.provider,
            config.debater_b.system_prompt,
            str(config.debater_b.temperature),
            str(config.debater_b.top_p),
            config.executor.name,
            config.executor.model,
            config.executor.provider,
            config.executor.system_prompt,
            str(config.executor.temperature),
            str(config.executor.top_p),
            str(config.max_turns),
            str(config.early_termination),
        ]
        return DebateCache._hash("|".join(parts))

    def _cache_key(self, query: str, config: DebateConfig) -> str:
        return self._hash(query) + ":" + self._config_fingerprint(config)

    def get(self, query: str, config: DebateConfig) -> tuple[list, int] | None:
        key = self._cache_key(query, config)

        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]

        qh = self._hash(query)
        ch = self._config_fingerprint(config)
        row = self._conn.execute(
            "SELECT transcript_json, max_turns FROM debates WHERE query_hash=? AND config_hash=?",
            (qh, ch),
        ).fetchone()
        if row is None:
            return None

        messages = ModelMessagesTypeAdapter.validate_json(row[0])
        result = (messages, row[1])

        self._memory_set(key, result)
        return result

    def set(
        self, query: str, config: DebateConfig, transcript: list
    ) -> None:
        key = self._cache_key(query, config)

        qh = self._hash(query)
        ch = self._config_fingerprint(config)
        transcript_json = ModelMessagesTypeAdapter.dump_json(transcript).decode()
        self._conn.execute(
            "INSERT OR REPLACE INTO debates (query_hash, config_hash, query, max_turns, transcript_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (qh, ch, query, config.max_turns, transcript_json, time.time()),
        )
        self._conn.commit()

        self._memory_set(key, (transcript, config.max_turns))

    def _memory_set(self, key: str, value: tuple[list, int]) -> None:
        if key in self._memory:
            self._memory.move_to_end(key)
        else:
            self._memory[key] = value
            while len(self._memory) > self._memory_max:
                self._memory.popitem(last=False)

    def close(self) -> None:
        self._conn.close()
