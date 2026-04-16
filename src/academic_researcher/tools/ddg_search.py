from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def ddg_web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the public web via DuckDuckGo (no API key).

    Returns a list of {title, href, snippet}.
    """
    try:
        from duckduckgo_search import DDGS
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "duckduckgo-search is required for ddg_web_search. "
            "Install dependencies from pyproject.toml."
        ) from e

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=max_results)):
            results.append(
                {
                    "rank": i + 1,
                    "title": r.get("title"),
                    "href": r.get("href"),
                    "snippet": r.get("body"),
                }
            )
    return results
