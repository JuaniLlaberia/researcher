import time
import requests
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.core.config import settings

SCHOOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,url"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = 2.0

def _get_with_retry(params: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
    """
    Get the search endpoint, retrying on 429 with (Retry-After or backoff).
    
    Args:
        params (Dict[str, Any]): Dictionary containing request parameters.
        headers (Dict[str, str]): Dictionary containing request headers.
    Returns:
        Response: Paper response in case it didn't fail (429) or didn't reach the max retries.
    """
    for attempt in range(_MAX_RETRIES):
        response = requests.get(SCHOOLAR_API, params=params, headers=headers, timeout=30)
        if response.status_code != 429 or attempt == _MAX_RETRIES - 1:
            response.raise_for_status()
            return response
        
        wait = float(response.headers.get("Retry-After", _BACKOFF_SECONDS * (attempt + 1)))
        time.sleep(wait)
    
    return response

def _resolve_pdf_url(paper: Dict[str, Any]) -> Optional[str]:
    """
    Find a downloadable PDF for a paper.

    Args:
        paper (Dict[str, Any]): Dictionary containing the paper's data.
    Returns:
        Optional[str]: It returns the url to download the paper or None in case it doesn't exists.
    """
    open_access = paper.get("openAccessPdf")
    if open_access and open_access.get("url"):
        return open_access["url"]

    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    return None

@tool
def search_schoolar(query: str, max_results: int = 5, open_access_only: bool = False) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar by keyword relevance. Returns paper metadata
    (title, authors, summary, abstract/pdf links) for ingestion.

    Args:
        query (str): Free-text search query (keywords, title fragment, topic).
        max_results (int): Maximum amount of papers to fetch. Default = 5.
        open_access_only (bool): If True, only return papers that expose an open-access PDF (i.e. papers that can actually be ingested).
    Returns:
        List[Dict[str, Any]]: List of paper dictionaries containing title, authors, summary, abstract_url, pdf_url. `pdf_url` may be None when no
        downloadable PDF is available.
    """
    if not query:
        return []

    params: Dict[str, Any] = {"query": query, "limit": max_results, "fields": FIELDS}
    if open_access_only:
        params["openAccessPdf"] = ""

    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    response = _get_with_retry(params, headers)
    data = response.json().get("data") or []

    results = []
    for paper in data:
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        if arxiv_id:
            source, external_id = "arxiv", arxiv_id
        else:
            source, external_id = "semantic_scholar", paper.get("paperId")

        results.append({
            "title": paper.get("title"),
            "authors": [a.get("name") for a in (paper.get("authors") or [])],
            "summary": paper.get("abstract") or "",
            "abstract_url": paper.get("url"),
            "pdf_url": _resolve_pdf_url(paper),
            "source": source,
            "external_id": external_id,
        })

    return results
