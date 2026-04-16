"""ArXiv search tool for academic research."""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from academic_researcher.net import sanitize_dead_local_proxies

# ── Process-level result cache ────────────────────────────────────────────────
# key: (query_lower, max_results)  →  raw list[dict]
# Prevents redundant HTTP requests when the LLM issues the same query twice.
_search_cache: dict[tuple[str, int], List[Dict[str, Any]]] = {}

_ATOM = "{http://www.w3.org/2005/Atom}"


def _arxiv_id(url: str) -> str:
    """Normalise an ArXiv URL/ID to a canonical form for deduplication.

    e.g. 'http://arxiv.org/abs/2303.08774v2' → '2303.08774'
    """
    # Strip version suffix and take only the numeric ID part
    base = url.rstrip("/").split("/")[-1]
    return base.split("v")[0]


def _fetch_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetch papers from ArXiv API, with process-level caching."""
    cache_key = (query.lower().strip(), max_results)
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    sanitize_dead_local_proxies()
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )

    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    papers: List[Dict[str, Any]] = []

    for entry in root.findall(f"{_ATOM}entry"):
        title_elem   = entry.find(f"{_ATOM}title")
        summary_elem = entry.find(f"{_ATOM}summary")
        published_el = entry.find(f"{_ATOM}published")
        id_elem      = entry.find(f"{_ATOM}id")

        if title_elem is None or summary_elem is None:
            continue

        authors = [
            name.text
            for author in entry.findall(f"{_ATOM}author")
            if (name := author.find(f"{_ATOM}name")) is not None and name.text
        ]

        raw_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

        papers.append({
            "title":     (title_elem.text or "").strip(),
            "authors":   authors,
            "summary":   (summary_elem.text or "").strip(),
            "published": (published_el.text or "")[:10],   # YYYY-MM-DD
            "arxiv_url": raw_url,
            "arxiv_id":  _arxiv_id(raw_url),
        })

    _search_cache[cache_key] = papers
    return papers


def _format_papers(papers: List[Dict[str, Any]], query: str) -> str:
    """Render a deduplicated paper list as a readable string."""
    if not papers:
        return f"No papers found for query: {query}"

    lines = [f"Found {len(papers)} papers for '{query}':\n"]
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += " et al."

        lines.append(f"{i}. **{p['title']}**")
        lines.append(f"   Authors  : {authors_str}")
        lines.append(f"   Published: {p['published']}")
        lines.append(f"   Summary  : {p['summary'][:400]}{'...' if len(p['summary']) > 400 else ''}")
        lines.append(f"   ArXiv    : {p['arxiv_url']}\n")

    return "\n".join(lines)


# ── Public helpers used by agent nodes ────────────────────────────────────────

def deduplicate_search_results(combined_text: str) -> str:
    """
    Merge multiple `=== arxiv_search ===` blocks and remove duplicate papers.

    Call this on the concatenated search_results string that comes out of the
    Researcher node after several tool calls.  Papers are considered duplicates
    when they share the same ArXiv ID (version-agnostic).

    Returns a single cleaned string with a dedup summary prepended.
    """
    # Split on section markers written by the agent nodes
    sections = [s.strip() for s in combined_text.split("=== arxiv_search ===") if s.strip()]
    if not sections:
        return combined_text

    seen_ids:  set[str]           = set()
    seen_urls: set[str]           = set()
    unique_lines: List[str]       = []
    total_seen  = 0
    total_kept  = 0

    for section in sections:
        paper_blocks = _split_into_paper_blocks(section)
        total_seen += len(paper_blocks)
        for block in paper_blocks:
            arxiv_id = _extract_arxiv_id_from_block(block)
            if arxiv_id and arxiv_id in seen_ids:
                continue
            if arxiv_id:
                seen_ids.add(arxiv_id)
            unique_lines.append(block)
            total_kept += 1

    removed = total_seen - total_kept
    header = (
        f"[Search results: {total_kept} unique papers"
        + (f", {removed} duplicate(s) removed" if removed else "")
        + "]\n"
    )
    return header + "\n---\n".join(unique_lines)


def _split_into_paper_blocks(text: str) -> List[str]:
    """Split a result section into individual paper blocks (split on numbered entries)."""
    import re
    # Each paper starts with a line like "1. **Title**"
    parts = re.split(r"(?=\n\d+\. \*\*)", "\n" + text)
    return [p.strip() for p in parts if p.strip()]


def _extract_arxiv_id_from_block(block: str) -> Optional[str]:
    """Pull the ArXiv ID out of a formatted paper block."""
    for line in block.splitlines():
        if "arxiv.org/abs/" in line.lower() or "arxiv_url" in line.lower():
            # e.g. "   ArXiv    : http://arxiv.org/abs/2303.08774v2"
            parts = line.split(":", 1)
            if len(parts) == 2:
                url = parts[1].strip()
                return _arxiv_id(url)
    return None


# ── LangChain Tool ─────────────────────────────────────────────────────────────

class ArXivSearchInput(BaseModel):
    """Input schema for ArXiv search tool."""
    query:       str = Field(description="Search query for ArXiv papers")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class ArXivSearchTool(BaseTool):
    """Tool for searching ArXiv papers, with built-in per-query deduplication."""

    name: str = "arxiv_search"
    description: str = (
        "Search for academic papers on ArXiv. "
        "Use this to find recent research papers related to your topic. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = ArXivSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            papers = _fetch_papers(query, max_results)

            # Deduplicate within this single result set (ArXiv API can return
            # the same paper under different versions for broad queries)
            seen: set[str] = set()
            unique: List[Dict[str, Any]] = []
            for p in papers:
                aid = p["arxiv_id"]
                if aid not in seen:
                    seen.add(aid)
                    unique.append(p)

            return _format_papers(unique, query)

        except Exception as exc:
            return f"Error searching ArXiv: {exc}"


def get_arxiv_search_tool() -> ArXivSearchTool:
    """Get an instance of the ArXiv search tool."""
    return ArXivSearchTool()
