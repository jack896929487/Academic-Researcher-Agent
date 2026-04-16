"""Structured logging and LangSmith tracing for the Academic Researcher Agent."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from academic_researcher.net import sanitize_dead_local_proxies


LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "agent_spans.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


class StructuredLogger:
    """
    Emits JSON-structured log lines to both stderr AND a local JSONL file
    so every run can be parsed by the analyze_logs script.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(console)

            # File handler (append-mode JSONL)
            _ensure_log_dir()
            fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(fh)

        self._log.setLevel(level)
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Return a new logger with extra context fields pre-attached."""
        child = StructuredLogger(self._log.name)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": level,
            "event": event,
            **self._context,
            **kwargs,
        }
        msg = json.dumps(record, ensure_ascii=False, default=str)
        getattr(self._log, level.lower(), self._log.info)(msg)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit("DEBUG", event, **kwargs)


class RunTracer:
    """
    Lightweight span-based tracer.  Each span records start time, end time,
    duration, and any attached metadata.  No external service needed.
    """

    def __init__(self, logger: StructuredLogger):
        self._logger = logger
        self._spans: list[Dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **meta: Any):
        """Context manager that times a block of code and logs the result."""
        start = time.perf_counter()
        span_record: Dict[str, Any] = {
            "span": name,
            "status": "ok",
            **meta,
        }
        try:
            yield span_record
        except Exception as exc:
            span_record["status"] = "error"
            span_record["error"] = str(exc)
            self._logger.error(f"span.{name}.error", **span_record)
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            span_record["elapsed_ms"] = elapsed_ms
            if span_record["status"] == "ok":
                self._logger.info(f"span.{name}.done", **span_record)
            self._spans.append(span_record)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all spans recorded in this trace."""
        total_ms = sum(s.get("elapsed_ms", 0) for s in self._spans)
        return {
            "total_spans": len(self._spans),
            "total_elapsed_ms": total_ms,
            "spans": self._spans,
        }


def setup_langsmith():
    """
    Enable LangSmith tracing if LANGCHAIN_API_KEY is present.
    Call this once at startup before any LLM calls.
    """
    sanitize_dead_local_proxies()
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if api_key and api_key != "your_langsmith_api_key":
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "academic-researcher-agent")
        return True
    return False


# Module-level singletons
agent_logger = StructuredLogger("academic_researcher")
