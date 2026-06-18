import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from langchain_core.tools import tool

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

def _arxiv_id(abstract_url: str) -> str:
    """
    Extract the bare arXiv id (no version) from an entry id URL.

    Args:
        abstract_url (str): Paper's url.
    Returns:
        str: Extracted arxiv id.
    """
    raw = abstract_url.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", raw)

@tool
def search_arxiv(
    title: str = "",
    author: str = "",
    keyword: str = "",
    max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv by title, author, and/or keyword. Returns paper metadata (title, authors, summary, abstract/pdf links) for ingestion.

    Args:
        title (str): Paper title.
        author (str): Paper's author name.
        keyword (str): Search keywords.
        max_results (int): Maximum amount of papers to fetch. Default = 5.
    Returns:
        List[Dict[str, Any]]: List of paper dictionaries containing title, authors, summary, pdf_url.
    """
    query_parts = []
    if title:
        query_parts.append(f"ti:{title}")
    if author:
        query_parts.append(f"au:{author}")
    if keyword:
        query_parts.append(f"all:{keyword}")
    if not query_parts:
        return []

    search_query = " AND ".join(query_parts)

    response = requests.get(
        ARXIV_API,
        params={"search_query": search_query, "max_results": max_results},
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    results = []
    for entry in root.findall("atom:entry", ATOM_NS):
        pdf_link = next(
            (l.get("href") for l in entry.findall("atom:link", ATOM_NS)
            if l.get("title") == "pdf"),
            None,
        )
        abstract_url = entry.find("atom:id", ATOM_NS).text
        results.append({
            "title": entry.find("atom:title", ATOM_NS).text.strip(),
            "authors": [
                a.find("atom:name", ATOM_NS).text
                for a in entry.findall("atom:author", ATOM_NS)
            ],
            "summary": entry.find("atom:summary", ATOM_NS).text.strip(),
            "abstract_url": abstract_url,
            "pdf_url": pdf_link,
            "source": "arxiv",
            "external_id": _arxiv_id(abstract_url),
          })

    return results