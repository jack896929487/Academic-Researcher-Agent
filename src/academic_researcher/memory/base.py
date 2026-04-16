"""Base classes for memory system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """A single memory entry."""
    
    id: Optional[str] = None
    user_id: str
    session_id: str
    content: str
    memory_type: str  # "research_topic", "user_preference", "search_result", etc.
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None


class BaseMemory(ABC):
    """Abstract base class for memory systems."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, 
                      user_id: str, 
                      session_id: Optional[str] = None,
                      memory_type: Optional[str] = None,
                      limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memory entries based on filters."""
        pass
    
    @abstractmethod
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def search(self, 
                    user_id: str, 
                    query: str, 
                    limit: int = 5) -> List[MemoryEntry]:
        """Search memory entries by content similarity."""
        pass