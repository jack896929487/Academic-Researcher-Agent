"""SQLite-based memory implementation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from academic_researcher.memory.base import BaseMemory, MemoryEntry


class SQLiteMemory(BaseMemory):
    """SQLite-based memory storage."""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memory_entries(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON memory_entries(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)")
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        if not entry.id:
            entry.id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_entries 
                (id, user_id, session_id, content, memory_type, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.user_id,
                entry.session_id,
                entry.content,
                entry.memory_type,
                json.dumps(entry.metadata),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat() if entry.updated_at else None
            ))
        
        return entry.id
    
    async def retrieve(self, 
                      user_id: str, 
                      session_id: Optional[str] = None,
                      memory_type: Optional[str] = None,
                      limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memory entries based on filters."""
        query = "SELECT * FROM memory_entries WHERE user_id = ?"
        params = [user_id]
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        entries = []
        for row in rows:
            entry = MemoryEntry(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                content=row["content"],
                memory_type=row["memory_type"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
            entries.append(entry)
        
        return entries
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ["content", "memory_type"]:
                set_clauses.append(f"{key} = ?")
                params.append(value)
            elif key == "metadata":
                set_clauses.append("metadata = ?")
                params.append(json.dumps(value))
        
        if not set_clauses:
            return False
        
        set_clauses.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(entry_id)
        
        query = f"UPDATE memory_entries SET {', '.join(set_clauses)} WHERE id = ?"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
    
    async def search(self, 
                    user_id: str, 
                    query: str, 
                    limit: int = 5) -> List[MemoryEntry]:
        """Search memory entries by content similarity (simple text matching)."""
        # Simple text search - in a production system, you'd use vector embeddings
        sql_query = """
            SELECT * FROM memory_entries 
            WHERE user_id = ? AND (
                content LIKE ? OR 
                memory_type LIKE ? OR
                metadata LIKE ?
            )
            ORDER BY created_at DESC 
            LIMIT ?
        """
        
        search_pattern = f"%{query}%"
        params = [user_id, search_pattern, search_pattern, search_pattern, limit]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query, params)
            rows = cursor.fetchall()
        
        entries = []
        for row in rows:
            entry = MemoryEntry(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                content=row["content"],
                memory_type=row["memory_type"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
            entries.append(entry)
        
        return entries
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT session_id) as total_sessions,
                    memory_type,
                    COUNT(*) as type_count
                FROM memory_entries 
                WHERE user_id = ?
                GROUP BY memory_type
            """, (user_id,))
            
            type_counts = {}
            total_entries = 0
            total_sessions = 0
            
            for row in cursor.fetchall():
                if total_entries == 0:  # First row
                    total_entries = row[0]
                    total_sessions = row[1]
                type_counts[row[2]] = row[3]
        
        return {
            "total_entries": total_entries,
            "total_sessions": total_sessions,
            "memory_types": type_counts
        }