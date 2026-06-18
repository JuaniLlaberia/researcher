from typing import Any, Dict, List

from src.researcher.tools.search_arxiv import search_arxiv
from src.researcher.tools.search_schoolar import search_schoolar
from src.researcher.tools.ingest_paper import IngestionPipeline

def scope_goal(research_goal: str, max_results: int = 5, ingest_cap: int = 3) -> int:
    """
    Search external sources for a research goal and ingest a few papers.

    Args:
        research_goal (str): The goal to scope.
        max_results (int): Max results to request from each source.
        ingest_cap (int): Max number of new papers to ingest.
    Returns:
        int: The number of papers actually ingested.
    """
    ingestion = IngestionPipeline()
    store = ingestion.store

    # Discover + dedup candidates across both sources.
    candidates: Dict[tuple, Dict[str, Any]] = {}
    for tool, kwargs in (
        (search_arxiv, {"keyword": research_goal, "max_results": max_results}),
        (search_schoolar, {"query": research_goal, "max_results": max_results}),
    ):
        try:
            results: List[Dict[str, Any]] = tool(**kwargs)
        except Exception as e:
            continue
        for r in results:
            key = (r.get("source"), r.get("external_id"))
            if key not in candidates and r.get("pdf_url") and r.get("external_id"):
                candidates[key] = r

    ingested = 0
    for c in candidates.values():
        if ingested >= ingest_cap:
            break
        source, external_id = c["source"], c["external_id"]
        try:
            if store.paper_exists(source, external_id):
                continue
            paper_meta = {
                "title": c["title"],
                "source": source,
                "external_id": external_id,
                "url": c.get("abstract_url"),
                "abstract": c.get("summary") or None,
                "authors": c.get("authors") or [],
            }
            ingestion.run(url=c["pdf_url"], paper_meta=paper_meta)
            ingested += 1
        except Exception as e:
            continue

    return ingested
