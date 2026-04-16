"""
Memory backend factory.

Reads the MEMORY_BACKEND environment variable and returns the
appropriate BaseMemory implementation:

    MEMORY_BACKEND=sqlite   →  SQLiteMemory   (default, keyword search)
    MEMORY_BACKEND=chroma   →  ChromaMemory   (semantic vector search)

Usage in agent code
-------------------
    from academic_researcher.memory.memory_factory import get_memory_backend

    memory = get_memory_backend()          # honours .env setting
    memory = get_memory_backend("chroma")  # force a specific backend

This keeps every agent's __init__ identical regardless of which
backend is active:

    self.memory = get_memory_backend()
    self.session_manager = SessionManager(self.memory)
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from academic_researcher.memory.base import BaseMemory


def get_memory_backend(
    backend: Optional[str] = None,
    *,
    db_path: str = "academic_agent_memory.db",
    chroma_dir: str = "chroma_memory",
) -> BaseMemory:
    """
    Return a configured memory backend instance.

    Parameters
    ----------
    backend : str | None
        "sqlite" or "chroma".  If None, reads MEMORY_BACKEND from .env
        (defaults to "sqlite" when the variable is absent).
    db_path : str
        SQLite database file path (used by both backends).
    chroma_dir : str
        Directory for ChromaDB persistence files.
    """
    load_dotenv(override=False)

    chosen = (backend or os.getenv("MEMORY_BACKEND", "sqlite")).lower().strip()

    if chosen == "chroma":
        try:
            from academic_researcher.memory.chroma_memory import ChromaMemory
            return ChromaMemory(persist_dir=chroma_dir, db_path=db_path)
        except ImportError as exc:
            print(
                f"[memory_factory] ChromaDB not available ({exc}). "
                "Falling back to SQLiteMemory. Run: pip install chromadb"
            )
            chosen = "sqlite"

    if chosen == "sqlite":
        from academic_researcher.memory.sqlite_memory import SQLiteMemory
        return SQLiteMemory(db_path=db_path)

    raise ValueError(
        f"Unknown MEMORY_BACKEND={chosen!r}. "
        "Valid options: 'sqlite' (default) or 'chroma'."
    )
