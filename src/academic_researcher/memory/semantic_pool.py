"""Chunked vector memory used for cross-session RAG recall."""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from academic_researcher.memory.chunking import chunk_text

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


def semantic_memory_available() -> bool:
    """Return True when ChromaDB is importable in the current environment."""
    return _CHROMA_AVAILABLE


@dataclass
class RetrievedChunk:
    """A semantically retrieved research chunk with source metadata."""

    chunk_id: str
    content: str
    topic: str
    goal: str
    memory_type: str
    session_id: str
    source_entry_id: str
    chunk_index: int
    distance: Optional[float] = None


class LangChainEmbeddingAdapter:
    """Adapter that makes LangChain embedding models compatible with Chroma."""

    def __init__(
        self,
        embeddings: Any,
        *,
        name: str,
        config: Optional[Dict[str, object]] = None,
    ):
        self._embeddings = embeddings
        self._name = name
        self._config = config or {}

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(list(input))

    def embed_query(self, input: Any) -> List[List[float]]:
        if isinstance(input, str):
            return [self._embeddings.embed_query(input)]
        return [self._embeddings.embed_query(text) for text in input]

    def name(self) -> str:
        return self._name

    @classmethod
    def build_from_config(cls, config: Dict[str, object]) -> "LangChainEmbeddingAdapter":
        raise NotImplementedError("Build from config is not supported for this adapter.")

    def get_config(self) -> Dict[str, object]:
        return dict(self._config)


class SentenceTransformersEmbeddingFunction:
    """Chroma-compatible wrapper around a local SentenceTransformer model."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        kwargs: Dict[str, object] = {}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        if device:
            kwargs["device"] = device

        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.device = device
        self._model = SentenceTransformer(model_name_or_path, **kwargs)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(input),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, input: Any) -> List[List[float]]:
        if isinstance(input, str):
            return [self.embed_documents([input])[0]]
        return self.embed_documents(list(input))

    def name(self) -> str:
        return "sentence_transformers_embedding"

    @classmethod
    def build_from_config(
        cls,
        config: Dict[str, object],
    ) -> "SentenceTransformersEmbeddingFunction":
        return cls(
            model_name_or_path=str(config["model_name_or_path"]),
            cache_folder=(
                str(config["cache_folder"])
                if config.get("cache_folder") is not None
                else None
            ),
            device=str(config["device"]) if config.get("device") is not None else None,
        )

    def get_config(self) -> Dict[str, object]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "cache_folder": self.cache_folder,
            "device": self.device,
        }


class LocalHashEmbeddingFunction:
    """
    Fully local embedding function for offline-friendly semantic retrieval.

    It is not as strong as a pretrained sentence embedding model, but it keeps
    the Chroma-backed vector workflow usable when the ONNX model cannot be
    downloaded in a restricted environment.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for document in input:
            vector = [0.0] * self.dimensions
            tokens = self._tokenize(document)
            if not tokens:
                tokens = ["__empty__"]

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign

            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            embeddings.append([value / norm for value in vector])

        return embeddings

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    @staticmethod
    def name() -> str:
        return "local_hash_embedding"

    @staticmethod
    def build_from_config(config: Dict[str, object]) -> "LocalHashEmbeddingFunction":
        dimensions = int(config.get("dimensions", 384))
        return LocalHashEmbeddingFunction(dimensions=dimensions)

    def get_config(self) -> Dict[str, object]:
        return {"dimensions": self.dimensions}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


class SemanticMemoryPool:
    """Persistent Chroma-backed store for chunked research artifacts."""

    def __init__(
        self,
        *,
        persist_dir: str = "chroma_memory",
        collection_name: str = "research_memory_chunks",
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        embedding_backend: str = "auto",
        embedding_model: Optional[str] = None,
        embedding_dimensions: int = 384,
        output_dimensions: Optional[int] = None,
        embedding_cache_dir: Optional[str] = None,
    ):
        if not _CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.output_dimensions = output_dimensions
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = self._build_embedding_function(embedding_cache_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def store_document(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        goal: str,
        memory_type: str,
        content: str,
        source_entry_id: Optional[str] = None,
    ) -> List[str]:
        """Chunk a document and persist each chunk into the vector index."""
        chunks = chunk_text(
            content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return []

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(self._build_document(topic=topic, goal=goal, content=chunk))
            metadatas.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "topic": topic[:300],
                    "goal": goal[:300],
                    "memory_type": memory_type,
                    "source_entry_id": (source_entry_id or "")[:300],
                    "chunk_index": chunk_index,
                }
            )

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return ids

    def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        memory_types: Optional[Iterable[str]] = None,
    ) -> List[RetrievedChunk]:
        """Return top semantic matches for a query."""
        requested = max(limit * 4, limit)
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(requested, max(1, self._collection.count())),
                where={"user_id": user_id},
            )
        except Exception:
            return []

        allowed_types = set(memory_types or [])
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if not docs:
            docs = [""] * len(ids)
        if not metas:
            metas = [{}] * len(ids)
        if not distances:
            distances = [None] * len(ids)

        retrieved: List[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, docs, metas, distances):
            metadata = metadata or {}
            memory_type = str(metadata.get("memory_type", ""))
            if allowed_types and memory_type not in allowed_types:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=self._extract_content(document),
                    topic=str(metadata.get("topic", "")),
                    goal=str(metadata.get("goal", "")),
                    memory_type=memory_type,
                    session_id=str(metadata.get("session_id", "")),
                    source_entry_id=str(metadata.get("source_entry_id", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=float(distance) if distance is not None else None,
                )
            )

            if len(retrieved) >= limit:
                break

        return retrieved

    @staticmethod
    def _build_document(*, topic: str, goal: str, content: str) -> str:
        return f"Topic: {topic}\nGoal: {goal}\n\n{content.strip()}"

    @staticmethod
    def _extract_content(document: str) -> str:
        marker = "\n\n"
        if marker in document:
            return document.split(marker, 1)[1].strip()
        return document.strip()

    def _configure_embedding_cache(self, embedding_cache_dir: Optional[str]) -> None:
        """
        Redirect Chroma's ONNX model cache into a writable project-local path.

        The default Chroma implementation writes to ~/.cache/chroma, which can be
        read-only in restricted environments. Keeping the cache under the project
        avoids permission issues and makes the dependency explicit.
        """
        cache_root = Path(embedding_cache_dir) if embedding_cache_dir else self.persist_dir / ".embedding_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        ONNXMiniLM_L6_V2.DOWNLOAD_PATH = cache_root / ONNXMiniLM_L6_V2.MODEL_NAME

    def _build_embedding_function(self, embedding_cache_dir: Optional[str]):
        backend = self._resolve_embedding_backend(embedding_cache_dir)
        self.embedding_backend = backend

        if backend in {"local", "local-hash", "hash"}:
            return LocalHashEmbeddingFunction(dimensions=self.embedding_dimensions)
        if backend in {"sentence-transformers", "sentence_transformer", "st"}:
            return self._build_sentence_transformers_embedding_function(
                embedding_cache_dir
            )
        if backend == "openai":
            return self._build_openai_embedding_function()
        if backend == "google":
            return self._build_google_embedding_function()
        if backend in {"default", "onnx", "onnx-default"}:
            self._configure_embedding_cache(embedding_cache_dir)
            return DefaultEmbeddingFunction()
        raise ValueError(
            f"Unknown embedding_backend={self.embedding_backend!r}. "
            "Use 'auto', 'sentence-transformers', 'openai', 'google', 'onnx', or 'local-hash'."
        )

    def _resolve_embedding_backend(self, embedding_cache_dir: Optional[str]) -> str:
        backend = self.embedding_backend.lower().strip()
        if backend != "auto":
            return backend

        local_model = self._default_local_sentence_transformer_model()
        if local_model is not None:
            self.embedding_model = str(local_model)
            return "sentence-transformers"

        chat_model = os.getenv("OPENAI_MODEL", "").lower()
        if "gemini" in chat_model and os.getenv("GOOGLE_API_KEY"):
            return "google"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GOOGLE_API_KEY"):
            return "google"
        if self.embedding_model and self._looks_like_local_model_path(self.embedding_model):
            return "sentence-transformers"
        if self._onnx_model_available(embedding_cache_dir):
            return "onnx"
        return "local-hash"

    def _build_openai_embedding_function(self) -> LangChainEmbeddingAdapter:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when SEMANTIC_EMBED_BACKEND=openai."
            )

        from langchain_openai import OpenAIEmbeddings

        model = self.embedding_model or os.getenv(
            "SEMANTIC_EMBED_MODEL",
            "text-embedding-3-small",
        )
        kwargs: Dict[str, object] = {
            "model": model,
            "api_key": api_key,
        }
        if self.output_dimensions is not None:
            kwargs["dimensions"] = self.output_dimensions

        embeddings = OpenAIEmbeddings(**kwargs)
        return LangChainEmbeddingAdapter(
            embeddings,
            name="openai_embedding",
            config={
                "backend": "openai",
                "model": model,
                "dimensions": self.output_dimensions,
            },
        )

    def _build_google_embedding_function(self) -> LangChainEmbeddingAdapter:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required when SEMANTIC_EMBED_BACKEND=google."
            )

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = self.embedding_model or os.getenv(
            "SEMANTIC_EMBED_MODEL",
            "models/text-embedding-004",
        )
        kwargs: Dict[str, object] = {
            "model": model,
            "api_key": api_key,
        }
        if self.output_dimensions is not None:
            kwargs["output_dimensionality"] = self.output_dimensions

        embeddings = GoogleGenerativeAIEmbeddings(**kwargs)
        return LangChainEmbeddingAdapter(
            embeddings,
            name="google_embedding",
            config={
                "backend": "google",
                "model": model,
                "dimensions": self.output_dimensions,
            },
        )

    def _build_sentence_transformers_embedding_function(
        self,
        embedding_cache_dir: Optional[str],
    ) -> SentenceTransformersEmbeddingFunction:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "sentence-transformers is not installed. "
                "Install it in the academic-agent environment first."
            ) from exc

        model_name_or_path = self.embedding_model or os.getenv(
            "SEMANTIC_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        resolved_local_model = self._resolve_local_model_path(str(model_name_or_path))
        if resolved_local_model is not None:
            model_name_or_path = str(resolved_local_model)

        local_default = self._default_local_sentence_transformer_model()
        if (
            local_default is not None
            and not self.embedding_model
            and not os.getenv("SEMANTIC_EMBED_MODEL")
        ):
            model_name_or_path = str(local_default)

        cache_root = (
            Path(embedding_cache_dir)
            if embedding_cache_dir
            else self.persist_dir / ".sentence_transformers_cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        device = os.getenv("SEMANTIC_EMBED_DEVICE")

        return SentenceTransformersEmbeddingFunction(
            model_name_or_path=model_name_or_path,
            cache_folder=str(cache_root),
            device=device,
        )

    def _onnx_model_available(self, embedding_cache_dir: Optional[str]) -> bool:
        cache_root = (
            Path(embedding_cache_dir)
            if embedding_cache_dir
            else self.persist_dir / ".embedding_cache"
        )
        onnx_root = cache_root / ONNXMiniLM_L6_V2.MODEL_NAME / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
        required_files = {
            "config.json",
            "model.onnx",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        }
        return all((onnx_root / name).exists() for name in required_files)

    @staticmethod
    def _looks_like_local_model_path(model_name_or_path: str) -> bool:
        return SemanticMemoryPool._resolve_local_model_path(model_name_or_path) is not None

    @classmethod
    def _default_local_sentence_transformer_model(cls) -> Optional[Path]:
        return cls._resolve_local_model_path("models/all-MiniLM-L6-v2")

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_local_model_path(cls, model_name_or_path: str) -> Optional[Path]:
        candidate = Path(model_name_or_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

        if candidate.is_absolute():
            return None

        repo_relative = cls._project_root() / candidate
        if repo_relative.exists():
            return repo_relative.resolve()

        return None
