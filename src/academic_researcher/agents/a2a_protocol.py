"""
A2A (Agent-to-Agent) message protocol.

Inspired by Google's Agent2Agent Protocol, this module defines a lightweight
message format that agents use to communicate within the multi-agent system.
Each message carries: sender, receiver, intent, payload, and optional metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    ORCHESTRATOR = "orchestrator"


class MessageIntent(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATE = "delegate"
    FEEDBACK = "feedback"
    COMPLETE = "complete"
    ERROR = "error"


class A2AMessage(BaseModel):
    """A single Agent-to-Agent message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentRole
    receiver: AgentRole
    intent: MessageIntent
    payload: Dict[str, Any]
    parent_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def reply(self, intent: MessageIntent, payload: Dict[str, Any]) -> "A2AMessage":
        """Create a reply message with roles swapped."""
        return A2AMessage(
            sender=self.receiver,
            receiver=self.sender,
            intent=intent,
            payload=payload,
            parent_id=self.id,
        )


class A2AMessageBus:
    """
    In-memory message bus for agent-to-agent communication.
    In production this could be backed by Redis, Kafka, etc.
    """

    def __init__(self):
        self._log: List[A2AMessage] = []

    def send(self, msg: A2AMessage) -> None:
        self._log.append(msg)

    def get_messages_for(self, role: AgentRole) -> List[A2AMessage]:
        return [m for m in self._log if m.receiver == role]

    def get_latest_for(self, role: AgentRole, intent: Optional[MessageIntent] = None) -> Optional[A2AMessage]:
        for msg in reversed(self._log):
            if msg.receiver == role:
                if intent is None or msg.intent == intent:
                    return msg
        return None

    def conversation_log(self) -> List[dict]:
        return [m.model_dump() for m in self._log]