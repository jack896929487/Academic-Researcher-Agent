"""PubMed search tool for biomedical literature."""

from __future__ import annotations

import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from academic_researcher.net import sanitize_dead_local_proxies


def _fetch_pubmed_ids(query: str, max_results: int) -> List[str]:
    sanitize_dead_local_proxies()
    encoded = urllib.parse.quote(query)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={encoded}&retmode=json&sort=relevance&retmax={max_results}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = resp.read().decode("utf-8")

    import json

    data = json.loads(payload)
    return list(data.get("esearchresult", {}).get("idlist", []))


def _safe_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _extract_pub_date(article: ET.Element) -> str:
    pub_date = article.find(".//PubDate")
    if pub_date is None:
        return ""

    year = _safe_text(pub_date.find("Year"))
    medline = _safe_text(pub_date.find("MedlineDate"))
    month = _safe_text(pub_date.find("Month"))
    day = _safe_text(pub_date.find("Day"))

    if year and month and day:
        return f"{year}-{month}-{day}"
    if year and month:
        return f"{year}-{month}"
    if year:
        return year
    return medline


def _fetch_pubmed_details(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []

    sanitize_dead_local_proxies()
    encoded_ids = ",".join(ids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={encoded_ids}&retmode=xml"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    papers: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_info = medline.find("Article") if medline is not None else None
        if medline is None or article_info is None:
            continue

        pmid = _safe_text(medline.find("PMID"))
        title = html.unescape(_safe_text(article_info.find("ArticleTitle")))

        abstract_parts = [
            _safe_text(node)
            for node in article_info.findall(".//Abstract/AbstractText")
            if _safe_text(node)
        ]
        abstract = " ".join(abstract_parts)

        authors = []
        for author in article_info.findall(".//AuthorList/Author"):
            last_name = _safe_text(author.find("LastName"))
            initials = _safe_text(author.find("Initials"))
            collective = _safe_text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
            elif last_name:
                authors.append(f"{last_name} {initials}".strip())

        journal = _safe_text(article_info.find(".//Journal/Title"))
        published = _extract_pub_date(article_info)

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "journal": journal,
                "published": published,
                "summary": abstract,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            }
        )

    return papers


def _format_papers(papers: List[Dict[str, Any]], query: str) -> str:
    if not papers:
        return f"No PubMed papers found for query: {query}"

    lines = [f"Found {len(papers)} PubMed papers for '{query}':\n"]
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " et al."

        lines.append(f"{i}. **{paper['title']}**")
        lines.append(f"   Authors  : {authors}")
        lines.append(f"   Journal  : {paper['journal']}")
        lines.append(f"   Published: {paper['published']}")
        lines.append(
            f"   Summary  : {paper['summary'][:500]}"
            f"{'...' if len(paper['summary']) > 500 else ''}"
        )
        lines.append(f"   PubMed   : {paper['pubmed_url']}\n")
    return "\n".join(lines)


class PubMedSearchInput(BaseModel):
    """Input schema for PubMed search tool."""

    query: str = Field(description="Search query for PubMed biomedical literature")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class PubMedSearchTool(BaseTool):
    """Tool for searching peer-reviewed biomedical literature on PubMed."""

    name: str = "pubmed_search"
    description: str = (
        "Search PubMed for peer-reviewed biomedical and clinical literature. "
        "Use this for biomedicine, biomarkers, diagnostics, oncology, genomics, "
        "or translational medicine topics."
    )
    args_schema: Type[BaseModel] = PubMedSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            ids = _fetch_pubmed_ids(query, max_results)
            papers = _fetch_pubmed_details(ids)
            return _format_papers(papers, query)
        except Exception as exc:
            return f"Error searching PubMed: {exc}"


def get_pubmed_search_tool() -> PubMedSearchTool:
    """Get an instance of the PubMed search tool."""
    return PubMedSearchTool()
