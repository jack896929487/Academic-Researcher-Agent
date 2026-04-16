"""
ChromaDB-based memory with semantic (vector) search.

Replaces the SQLite LIKE-based search with cosine similarity over text embeddings,
so queries like "transformer architecture" will recall entries about
"attention mechanism" or "BERT fine-tuning" that are semantically related.

Architecture
------------
- ChromaDB stores the *vector index* (embeddings + ids)
- SQLite (via SQLiteMemory) stores the *full structured records*
  (user_id, session_id, memory_type, metadata, timestamps)

Both stores are written together on every `store()` call and the
`search()` method uses ChromaDB for recall then fetches full records
from SQLite, combining the best of both worlds.

Embedding model
---------------
Uses ChromaDB's built-in `DefaultEmbeddingFunction` (all-MiniLM-L6-v2
via onnxruntime, downloaded once to ~/.cache/chroma/).
No OpenAI / Google API calls are needed for embeddings.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

from academic_researcher.memory.base import BaseMemory, MemoryEntry


def _require_chroma() -> None:
    if not _CHROMA_AVAILABLE:
        raise ImportError(
            "chromadb is not installed. Run: pip install chromadb"
        )


class ChromaMemory(BaseMemory):
    """
    Hybrid memory backend:
      - ChromaDB  →  semantic search  (find by meaning)
      - SQLite    →  structured CRUD  (filter by user/type/time)

    Parameters
    ----------
    persist_dir : str
        Directory where ChromaDB persists its index files.
    db_path : str
        Path to the SQLite file for structured metadata.
    collection_name : str
        Name of the ChromaDB collection (one per project is fine).
    """

    def __init__(
        self,
        persist_dir: str = "chroma_memory",
        db_path: str = "academic_agent_memory.db",
        collection_name: str = "research_memory",
    ):
        _require_chroma()

        self._db_path = Path(db_path)
        self._init_sqlite()

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ─────────────────────────────── SQLite helpers ──────────────────────────

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id           TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    memory_type  TEXT NOT NULL,
                    metadata     TEXT,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT
                )
            """)
            for col, idx in [
                ("user_id",     "idx_ce_user"),
                ("session_id",  "idx_ce_session"),
                ("memory_type", "idx_ce_type"),
                ("created_at",  "idx_ce_created"),
            ]:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} "
                    f"ON memory_entries({col})"
                )

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=(
                datetime.fromisoformat(row["updated_at"])
                if row["updated_at"] else None
            ),
        )

    def _fetch_by_ids(self, ids: List[str]) -> List[MemoryEntry]:
        """Return MemoryEntry list in the same order as `ids`."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_entries WHERE id IN ({placeholders})",
                ids,
            ).fetchall()

        by_id = {r["id"]: self._row_to_entry(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # ─────────────────────────────── BaseMemory API ──────────────────────────

    async def store(self, entry: MemoryEntry) -> str:
        """Persist entry in both SQLite (metadata) and ChromaDB (vector)."""
        if not entry.id:
            entry.id = str(uuid.uuid4())

        # ── 1. SQLite ─────────────────────────────────────────────────────
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entries
                (id, user_id, session_id, content, memory_type,
                 metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.user_id,
                    entry.session_id,
                    entry.content,
                    entry.memory_type,
                    json.dumps(entry.metadata),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat() if entry.updated_at else None,
                ),
            )

        # ── 2. ChromaDB (upsert so re-runs don't duplicate) ───────────────
        chroma_meta = {
            "user_id":     entry.user_id,
            "session_id":  entry.session_id,
            "memory_type": entry.memory_type,
            "created_at":  entry.created_at.isoformat(),
        }
        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[chroma_meta],
        )

        return entry.id

    async def retrieve(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Structured retrieval from SQLite (exact filters)."""
        query = "SELECT * FROM memory_entries WHERE user_id = ?"
        params: list = [user_id]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(r) for r in rows]

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update structured fields in SQLite only."""
        set_clauses: list[str] = []
        params: list = []

        for key, value in updates.items():
            if key in ("content", "memory_type"):
                set_clauses.append(f"{key} = ?")
                params.append(value)
            elif key == "metadata":
                set_clauses.append("metadata = ?")
                params.append(json.dumps(value))

        if not set_clauses:
            return False

        set_clauses.append("updated_at = ?")
        params.extend([datetime.now().isoformat(), entry_id])

        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                f"UPDATE memory_entries SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
            updated = cur.rowcount > 0

        # Keep ChromaDB document in sync if content changed
        if updated and "content" in updates:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
                ).fetchone()
            if row:
                entry = self._row_to_entry(row)
                self._collection.upsert(
                    ids=[entry_id],
                    documents=[entry.content],
                    metadatas=[{
                        "user_id":     entry.user_id,
                        "session_id":  entry.session_id,
                        "memory_type": entry.memory_type,
                        "created_at":  entry.created_at.isoformat(),
                    }],
                )

        return updated

    async def delete(self, entry_id: str) -> bool:
        """Remove from both stores."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "DELETE FROM memory_entries WHERE id = ?", (entry_id,)
            )
            deleted = cur.rowcount > 0

        if deleted:
            try:
                self._collection.delete(ids=[entry_id])
            except Exception:
                pass

        return deleted

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """
        Semantic search using ChromaDB vector similarity.

        Steps
        -----
        1. Embed `query` with the same model used during `store()`
        2. Query ChromaDB for top-k nearest neighbours filtered by user_id
        3. Fetch full MemoryEntry records from SQLite by the returned ids
        """
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(limit, max(1, self._collection.count())),
                where={"user_id": user_id},
            )
        except Exception:
            # ChromaDB raises if collection is empty or filter matches nothing
            return []

        ids: List[str] = results.get("ids", [[]])[0]
        return self._fetch_by_ids(ids)

    # ─────────────────────────────── Extras ──────────────────────────────────

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Return memory statistics (same interface as SQLiteMemory)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT memory_type, COUNT(*) as cnt
                FROM memory_entries
                WHERE user_id = ?
                GROUP BY memory_type
                """,
                (user_id,),
            ).fetchall()

            total = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT session_id) "
                "FROM memory_entries WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        return {
            "total_entries":  total[0] if total else 0,
            "total_sessions": total[1] if total else 0,
            "memory_types":   {r[0]: r[1] for r in rows},
            "backend":        "chromadb+sqlite",
        }
