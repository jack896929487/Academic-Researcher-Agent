from __future__ import annotations

from langchain_core.tools import tool

from academic_researcher.memory.sqlite_store import SQLitePreferenceStore


@tool
def remember_user_fact(user_id: str, fact: str) -> str:
    """Store a durable user preference/fact for personalization (SQLite).

    This is a teaching tool: keep facts short and non-sensitive.
    """
    store = SQLitePreferenceStore()
    store.add_fact(user_id=user_id, fact=fact)
    return "Saved."


@tool
def recall_user_facts(user_id: str, limit: int = 20) -> list[str]:
    """Recall stored facts for a user (most recent first)."""
    store = SQLitePreferenceStore()
    return store.list_facts(user_id=user_id, limit=limit)
