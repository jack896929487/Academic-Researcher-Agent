from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver


@dataclass
class SessionMemory:
    """Short-term memory via LangGraph checkpointer (thread_id scoped)."""

    checkpointer: MemorySaver

    @staticmethod
    def new_in_memory() -> "SessionMemory":
        return SessionMemory(checkpointer=MemorySaver())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
