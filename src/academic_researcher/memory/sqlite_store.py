from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SQLitePreferenceStore:
    """Very small long-term memory store (preferences/facts).

    This is intentionally boring SQLite so you can inspect the DB file directly.
    """

    db_path: Path = Path(".agent_memory.sqlite3")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_facts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT NOT NULL,
              fact TEXT NOT NULL,
              created_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
        return conn

    def add_fact(self, user_id: str, fact: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO user_facts(user_id, fact, created_at) VALUES (?, ?, ?);",
            (user_id, fact, int(time.time())),
        )
        conn.commit()
        conn.close()

    def list_facts(self, user_id: str, limit: int = 20) -> list[str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT fact FROM user_facts WHERE user_id = ? ORDER BY id DESC LIMIT ?;",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
