"""Helpers for assembling domain-aware research tool sets."""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.tools import BaseTool

from academic_researcher.tools.arxiv_search import get_arxiv_search_tool
from academic_researcher.tools.mcp_tools import get_demo_mcp_tools, get_mcp_tools
from academic_researcher.tools.pubmed_search import get_pubmed_search_tool


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_research_tools(
    domain: Optional[str] = None,
    *,
    include_demo_mcp: Optional[bool] = None,
) -> List[BaseTool]:
    """Return a domain-aware ordered list of literature search tools."""

    domain_key = (domain or "").strip().lower()
    if domain_key == "biomedicine":
        tools: List[BaseTool] = [get_pubmed_search_tool(), get_arxiv_search_tool()]
    else:
        tools = [get_arxiv_search_tool(), get_pubmed_search_tool()]

    mcp_tools = get_mcp_tools()
    if mcp_tools:
        tools.extend(mcp_tools)
    else:
        if include_demo_mcp is None:
            include_demo_mcp = _env_flag("ENABLE_DEMO_MCP_TOOLS", "false")
        if include_demo_mcp:
            tools.extend(get_demo_mcp_tools())

    deduped: List[BaseTool] = []
    seen_names = set()
    for tool in tools:
        if tool.name in seen_names:
            continue
        seen_names.add(tool.name)
        deduped.append(tool)
    return deduped
