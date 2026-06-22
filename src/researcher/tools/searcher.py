from typing import Any, Callable, Dict, List, Optional

from src.core.logging import get_logger, log_stage
from src.researcher.models import EvidenceItem
from src.db.vector_store import VectorStore
from .ingest_paper import IngestionPipeline
from .search_arxiv import search_arxiv
from .search_schoolar import search_schoolar

DISCOVERY_VARIANTS = 2
DEFAULT_INGEST_CAP = 3
# Max NEW papers ingested per run. Once hit, gather() stops downloading and just
# retrieves from the corpus already accumulated this run. Bounds worst-case time.
INGESTION_BUDGET = 15

log = get_logger("searcher")

# A gate decides which discovered candidates are worth ingesting.
Gate = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]

class Searcher:
    """
    Agent-agnostic paper search/ingest/retrieve service.

    Composes the external search tools, the ingestion pipeline, and the vector
    store so every agent that needs literature (Literature Reviewer, Hypothesizer,
    Critic) shares one implementation instead of duplicating it.
    """
    def __init__(self, vector_store: Optional[VectorStore] = None, ingestion: Optional[IngestionPipeline] = None) -> None:
        """
        Initializes the Searcher. Builds its own ingestion pipeline and vector
        store by default, or accepts shared instances to avoid loading the
        (heavy) Docling converter / reranker more than once per process.

        Args:
            vector_store (Optional[VectorStore]): Shared vector store, if any.
            ingestion (Optional[IngestionPipeline]): Shared ingestion pipeline, if any.
        """
        self.ingestion = ingestion or IngestionPipeline()
        self.vector_store = vector_store or self.ingestion.store
        # New ingestions performed this run; bounded by INGESTION_BUDGET.
        self._ingested_count = 0

    def reset_ingestion_budget(self) -> None:
        """
        Reset the per-run ingestion counter. Called by the orchestrator at the start
        of each run, since the Searcher is shared and long-lived but the budget is
        per-run.
        """
        self._ingested_count = 0

    def gather(
        self,
        main_query: str,
        queries: List[str],
        gate: Optional[Gate] = None,
        ingest_cap: int = DEFAULT_INGEST_CAP,
    ) -> List[EvidenceItem]:
        """
        Full search pass: discover candidates, select them (via the injected gate
        or a plain cap), ingest the selection, then hybrid-retrieve evidence.

        Args:
            main_query (str): Primary query (keyword + rerank anchor).
            queries (List[str]): Semantic query variants.
            gate (Optional[Gate]): Selects candidates to ingest. None -> first `ingest_cap`.
            ingest_cap (int): Cap used when no gate is provided.
        Returns:
            List[EvidenceItem]: Evidence retrieved after ingestion.
        """
        candidates = self._discover(main_query, queries)
        selected = gate(candidates) if gate else candidates[:ingest_cap]
        log.info("gather: %d candidates discovered, %d selected for ingestion", len(candidates), len(selected))
        self._ingest_selected(selected)
        return self.retrieve(main_query, queries)

    @staticmethod
    def merge_evidence(existing: List[EvidenceItem], new: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Merge newly retrieved evidence into the evidence accumulated so far,
        deduplicating by chunk_id (falling back to (paper_id, text) when a chunk
        id is unavailable). Existing items are preserved; duplicates are dropped.

        Args:
            existing (List[EvidenceItem]): Evidence already in state.
            new (List[EvidenceItem]): Evidence from the latest retrieval.
        Returns:
            List[EvidenceItem]: Deduplicated union of both lists.
        """
        def key(item: EvidenceItem):
            return item.chunk_id or (item.paper_id, item.text)

        merged = list(existing)
        seen = {key(item) for item in existing}
        for item in new:
            k = key(item)
            if k not in seen:
                seen.add(k)
                merged.append(item)
        return merged

    def retrieve(self, main_query: str, queries: List[str]) -> List[EvidenceItem]:
        """
        Hybrid-retrieve full-text chunks and normalize into EvidenceItems.

        Args:
            main_query (str): Main hypothesis query.
            queries (List[str]): List of query variants.
        Returns:
            List[EvidenceItem]: List of formatted evidence objects (empty on failure).
        """
        try:
            results = self.vector_store.retrieve_documents(main_query=main_query, queries=queries)
        except Exception:
            log.exception("retrieve failed for %r — returning no evidence", main_query)
            return []

        log.info("retrieve: %d chunks for %r", len(results), main_query)

        return [
            EvidenceItem(
                source_type="vector_db",
                chunk_id=r.get("id"),
                title=r["metadata"].get("title"),
                url=r["metadata"].get("url"),
                text=r["text"],
                paper_id=r["metadata"].get("paper_id"),
                score=r.get("score"),
            )
            for r in results
        ]

    def _discover(self, main_query: str, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Search arXiv + Semantic Scholar; merge + dedup ingestable candidates.

        Args:
            main_query (str): Main hypothesis query.
            queries (List[str]): List of query variants.
        Returns:
            List[Dict[str, Any]]: Candidate paper dicts.
        """
        terms = [main_query] + queries[:DISCOVERY_VARIANTS]
        candidates: Dict[tuple, Dict[str, Any]] = {}

        for term in terms:
            for tool, source, kwargs in (
                (search_arxiv, "arxiv", {"keyword": term, "max_results": 5}),
                (search_schoolar, "semantic_scholar", {"query": term, "max_results": 5}),
            ):
                try:
                    results = tool(**kwargs)
                except Exception:
                    log.warning("%s search failed for %r", source, term)
                    continue
                log.debug("%s returned %d results for %r", source, len(results), term)
                for r in results:
                    key = (r.get("source"), r.get("external_id"))

                    if key not in candidates and r.get("pdf_url") and r.get("external_id"):
                        candidates[key] = r

        return list(candidates.values())

    def _ingest_selected(self, selected: List[Dict[str, Any]]) -> None:
        """
        Ingest each selected paper, skipping any already stored (dedup).

        Args:
            selected (List[Dict[str, Any]]): List of papers to ingest.
        Returns:
            None.
        """
        for c in selected:
            source, external_id = c["source"], c["external_id"]
            title = c.get("title", "")
            try:
                if self.vector_store.paper_exists(source, external_id):
                    log.info("skip (already stored): [%s] %s", source, title[:70])
                    continue
                if self._ingested_count >= INGESTION_BUDGET:
                    log.info("ingestion budget (%d) reached — retrieving from stored corpus only", INGESTION_BUDGET)
                    break
                paper_meta = {
                    "title": c["title"],
                    "source": source,
                    "external_id": external_id,
                    "url": c.get("abstract_url"),
                    "abstract": c.get("summary") or None,
                    "authors": c.get("authors") or [],
                }
                with log_stage(log, f"ingest [{source}] {title[:60]}"):
                    self.ingestion.run(url=c["pdf_url"], paper_meta=paper_meta)
                self._ingested_count += 1
            except Exception:
                log.exception("ingestion failed: [%s] %s", source, title[:70])
                continue
