from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage


class ResearchRequest:
    """
    User-facing request for an academic research run.

    Note: we keep this as a plain class to avoid extra dependencies early in Day 1.
    """

    def __init__(
        self,
        topic: str,
        goal: str,
        user_id: str = "default",
        domain: Optional[str] = None,
    ):
        self.topic = topic
        self.goal = goal
        self.user_id = user_id
        self.domain = domain


class ResearchState(dict):
    """
    LangGraph state container.

    We store fields in a dict so LangGraph can serialize/merge it easily.
    """

    topic: str
    goal: str
    user_id: str
    domain: Optional[str]
    messages: List[BaseMessage]
    plan: Optional[str]
    report: Optional[str]

