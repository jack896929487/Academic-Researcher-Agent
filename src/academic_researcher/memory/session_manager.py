"""Session management and user preferences."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from academic_researcher.memory.base import BaseMemory, MemoryEntry
from academic_researcher.memory.semantic_pool import SemanticMemoryPool, RetrievedChunk

_SEMANTIC_MEMORY_TYPES = {"search_results_chunk", "research_report_chunk"}


class SessionManager:
    """Manages structured session history plus chunked semantic recall."""

    def __init__(
        self,
        memory: BaseMemory,
        semantic_memory: Optional[SemanticMemoryPool] = None,
    ):
        load_dotenv(override=False)
        self.memory = memory
        self.semantic_memory = semantic_memory or self._build_semantic_memory()

    def create_session(self, user_id: str) -> str:
        """Create a new session for a user."""
        return f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

    async def store_user_preference(
        self,
        user_id: str,
        session_id: str,
        preference_type: str,
        preference_value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a user preference."""
        entry = MemoryEntry(
            user_id=user_id,
            session_id=session_id,
            content=preference_value,
            memory_type=f"user_preference_{preference_type}",
            metadata=metadata or {},
            created_at=datetime.now(),
        )
        return await self.memory.store(entry)

    async def store_research_context(
        self,
        user_id: str,
        session_id: str,
        topic: str,
        goal: str,
        search_results: Optional[str] = None,
        report: Optional[str] = None,
    ) -> List[str]:
        """
        Store research context from a session.

        Structured records stay in the primary memory backend. Long-form search
        results and reports are also chunked into the semantic pool for future RAG.
        """
        entry_ids: List[str] = []

        topic_entry = MemoryEntry(
            user_id=user_id,
            session_id=session_id,
            content=topic,
            memory_type="research_topic",
            metadata={"goal": goal},
            created_at=datetime.now(),
        )
        entry_ids.append(await self.memory.store(topic_entry))

        if search_results:
            search_entry = MemoryEntry(
                user_id=user_id,
                session_id=session_id,
                content=search_results,
                memory_type="search_results",
                metadata={"topic": topic},
                created_at=datetime.now(),
            )
            search_entry_id = await self.memory.store(search_entry)
            entry_ids.append(search_entry_id)
            entry_ids.extend(
                await self._store_semantic_document(
                    user_id=user_id,
                    session_id=session_id,
                    topic=topic,
                    goal=goal,
                    memory_type="search_results_chunk",
                    content=search_results,
                    source_entry_id=search_entry_id,
                )
            )

        if report:
            report_entry = MemoryEntry(
                user_id=user_id,
                session_id=session_id,
                content=report,
                memory_type="research_report",
                metadata={"topic": topic, "goal": goal},
                created_at=datetime.now(),
            )
            report_entry_id = await self.memory.store(report_entry)
            entry_ids.append(report_entry_id)
            entry_ids.extend(
                await self._store_semantic_document(
                    user_id=user_id,
                    session_id=session_id,
                    topic=topic,
                    goal=goal,
                    memory_type="research_report_chunk",
                    content=report,
                    source_entry_id=report_entry_id,
                )
            )

        return entry_ids

    async def get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """Get all user preferences."""
        entries = await self.memory.retrieve(user_id=user_id, limit=200)

        preferences: Dict[str, str] = {}
        for entry in entries:
            if not entry.memory_type.startswith("user_preference_"):
                continue
            pref_type = entry.memory_type.replace("user_preference_", "", 1)
            if pref_type not in preferences:
                preferences[pref_type] = entry.content

        return preferences

    async def get_research_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's research history."""
        entries = await self.memory.retrieve(
            user_id=user_id,
            memory_type="research_topic",
            limit=limit,
        )

        history = []
        for entry in entries:
            history.append(
                {
                    "session_id": entry.session_id,
                    "topic": entry.content,
                    "goal": entry.metadata.get("goal", ""),
                    "created_at": entry.created_at.isoformat(),
                }
            )

        return history

    async def get_relevant_context(
        self,
        user_id: str,
        current_topic: str,
        limit: int = 5,
    ) -> str:
        """
        Get retrieval-augmented context from prior sessions.

        Priority:
        1. semantic recall over chunked search results and reports
        2. fallback structured history if semantic recall is unavailable or empty
        """
        semantic_chunks = await self._semantic_search(
            user_id=user_id,
            query=current_topic,
            limit=limit,
        )

        if semantic_chunks:
            return self._render_semantic_context(semantic_chunks)

        relevant_entries = await self.memory.search(
            user_id=user_id,
            query=current_topic,
            limit=limit,
        )
        if not relevant_entries:
            return ""

        context_parts = ["Previous research history:"]
        for entry in relevant_entries:
            if entry.memory_type == "research_topic":
                goal = entry.metadata.get("goal", "")
                context_parts.append(f"- Topic: {entry.content} | Goal: {goal}")
            elif entry.memory_type == "user_preference_research_domain":
                context_parts.append(f"- Preferred research domain: {entry.content}")
            elif entry.memory_type == "user_preference_report_style":
                context_parts.append(f"- Preferred report style: {entry.content}")

        return "\n".join(context_parts)

    async def update_user_preference(
        self,
        user_id: str,
        preference_type: str,
        new_value: str,
    ) -> bool:
        """Update an existing user preference."""
        entries = await self.memory.retrieve(
            user_id=user_id,
            memory_type=f"user_preference_{preference_type}",
            limit=1,
        )

        if entries:
            return await self.memory.update(entries[0].id, {"content": new_value})

        await self.store_user_preference(
            user_id=user_id,
            session_id="preference_update",
            preference_type=preference_type,
            preference_value=new_value,
        )
        return True

    def _build_semantic_memory(self) -> Optional[SemanticMemoryPool]:
        enabled = os.getenv("SEMANTIC_MEMORY_ENABLED", "true").lower()
        if enabled in {"0", "false", "no"}:
            return None

        output_dimensions = os.getenv("SEMANTIC_EMBED_OUTPUT_DIMENSIONS")

        try:
            return SemanticMemoryPool(
                persist_dir=os.getenv("SEMANTIC_MEMORY_DIR", "chroma_memory"),
                collection_name=os.getenv(
                    "SEMANTIC_MEMORY_COLLECTION",
                    "research_memory_chunks",
                ),
                chunk_size=int(os.getenv("SEMANTIC_CHUNK_SIZE", "1200")),
                chunk_overlap=int(os.getenv("SEMANTIC_CHUNK_OVERLAP", "150")),
                embedding_backend=os.getenv("SEMANTIC_EMBED_BACKEND", "auto"),
                embedding_model=os.getenv("SEMANTIC_EMBED_MODEL"),
                embedding_dimensions=int(os.getenv("SEMANTIC_EMBED_DIMENSIONS", "384")),
                output_dimensions=(
                    int(output_dimensions) if output_dimensions else None
                ),
                embedding_cache_dir=os.getenv("SEMANTIC_EMBED_CACHE_DIR"),
            )
        except Exception as exc:
            requested_backend = os.getenv("SEMANTIC_EMBED_BACKEND", "auto").lower()
            if requested_backend not in {
                "auto",
                "local",
                "local-hash",
                "hash",
                "sentence-transformers",
                "sentence_transformer",
                "st",
            }:
                raise ValueError(
                    f"Failed to initialize semantic memory backend {requested_backend!r}: {exc}"
                ) from exc
            return None

    async def _store_semantic_document(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        goal: str,
        memory_type: str,
        content: str,
        source_entry_id: str,
    ) -> List[str]:
        if self.semantic_memory is None:
            return []

        return await asyncio.to_thread(
            self.semantic_memory.store_document,
            user_id=user_id,
            session_id=session_id,
            topic=topic,
            goal=goal,
            memory_type=memory_type,
            content=content,
            source_entry_id=source_entry_id,
        )

    async def _semantic_search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
    ) -> List[RetrievedChunk]:
        if self.semantic_memory is None:
            return []

        results = await asyncio.to_thread(
            self.semantic_memory.search,
            user_id=user_id,
            query=query,
            limit=max(limit * 2, limit),
            memory_types=_SEMANTIC_MEMORY_TYPES,
        )

        deduped: List[RetrievedChunk] = []
        seen_sources = set()
        for chunk in results:
            source_key = (chunk.source_entry_id or chunk.session_id, chunk.memory_type)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            deduped.append(chunk)
            if len(deduped) >= limit:
                break

        return deduped

    @staticmethod
    def _render_semantic_context(chunks: List[RetrievedChunk]) -> str:
        parts = ["Relevant historical research memory:"]
        for index, chunk in enumerate(chunks, start=1):
            source_label = chunk.memory_type.replace("_chunk", "").replace("_", " ")
            snippet = chunk.content.strip()
            if len(snippet) > 420:
                snippet = snippet[:420].rstrip() + "..."

            header = (
                f"{index}. [{source_label}] Topic: {chunk.topic or 'unknown'}"
            )
            if chunk.goal:
                header += f" | Goal: {chunk.goal}"

            parts.append(header)
            parts.append(f"   {snippet}")

        return "\n".join(parts)
