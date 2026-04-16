from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage


_URL_RE = re.compile(r"https?://\\S+")


@dataclass(frozen=True)
class EvalResult:
    ok: bool
    reasons: list[str]


def basic_academic_answer_checks(answer_text: str) -> EvalResult:
    """Heuristic checks that catch common agent failure modes (cheap, offline)."""

    reasons: list[str] = []
    t = answer_text.strip()

    if len(t) < 120:
        reasons.append("Answer too short for an academic briefing.")

    if not re.search(r"(?i)\\b(limitation|limitations|limitation:)\\b", t):
        reasons.append("Missing an explicit limitations section (word 'limitations' not found).")

    if len(_URL_RE.findall(t)) < 1:
        reasons.append("No explicit URLs found; ensure references include links when possible.")

    return EvalResult(ok=len(reasons) == 0, reasons=reasons)


def last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content
    return ""
