from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.researcher.tools.search_arxiv import search_arxiv
from src.researcher.tools.search_schoolar import search_schoolar
from src.researcher.tools.ingest_paper import IngestionPipeline
from .utils.models import (
    Finding,
    DataValidatorOutput,
    QueriesOutput,
    FindingExtractorOutput,
    EvidenceItem,
    RelevanceGateOutput,
)
from .utils.prompts import (
    QUERIES_GENERATION_PROMPT,
    DATA_VALIDATION_PROMPT,
    FINDING_EXTRATOR_PROMPT,
    RELEVANCE_GATE_PROMPT,
)

MAX_RETRIES_PER_NODE = 3
INGEST_CAP = 3
DISCOVERY_VARIANTS = 2

class LiteratureReviewerState(TypedDict):
    research_goal: str
    hypothesis: str

    main_query: str
    queries: List[str]
    queries_feedback: str
    raw_data: List[EvidenceItem]
    has_sufficient_data: Literal["sufficient", "insufficient", "needs_different_queries"]
    findings: List[Finding]

    query_gen_retries: int
    search_retries: int

class LiteratureReviewer:
    """
    Literature reviewer agent. Given a goal and a hypothesis it find and analyzes the most relevant papers.
    """
    def __init__(self) -> None:
        """
        Initializes the LiteratureReviwer class.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        ))

        self.ingestion = IngestionPipeline()
        self.vector_store = self.ingestion.store

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Builds and compiles the agent graph.

        Returns:
            StateGraph: The compiled LangGraph graph.
        """
        graph = StateGraph(LiteratureReviewerState)
        graph.add_node("queries_generator", self._queries_generator_node)
        graph.add_node("queries_validator", self._queries_validator)
        graph.add_node("searcher", self._searcher_node)
        graph.add_node("data_evaluator", self._data_evaluator_node)
        graph.add_node("data_router", self._data_router)
        graph.add_node("finding_extractor", self._finding_extractor_node)

        graph.set_entry_point("queries_generator")
        graph.add_edge("queries_generator", "queries_validator")
        graph.add_edge("searcher", "data_evaluator")
        graph.add_edge("data_evaluator", "data_router")
        graph.set_finish_point("finding_extractor")

        return graph.compile()

    def _queries_generator_node(self, state: LiteratureReviewerState) -> Dict[str, Any]:
        """
        Generate one main query plus N variants from goal + hypothesis.

        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=QUERIES_GENERATION_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypothesis": state["hypothesis"],
                    "feedback": state["queries_feedback"],
                },
                output_schema=QueriesOutput,
            )
            data = result if isinstance(result, QueriesOutput) else QueriesOutput(**result.model_dump())
            return {"main_query": data.main_query, "queries": data.queries}
        
        except Exception:
            return {"main_query": "", "queries": []}

    def _queries_validator(self, state: LiteratureReviewerState) -> Command[Literal["searcher", "queries_generator"]]:
        """
        Route to the searcher if queries are valid, else regenerate (capped).

        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            Command: Langgraph function that redirects to specified node in the graph.
        """
        if len(state["queries"]) >= 1 and state["main_query"] != "":
            return Command(update={"query_gen_retries": 0}, goto="searcher")

        if state["query_gen_retries"] + 1 >= MAX_RETRIES_PER_NODE:
            return Command(goto=END)

        return Command(
            update={
                "query_gen_retries": state["query_gen_retries"] + 1,
                "queries_feedback": "Must produce at least 1 query and a non-empty main_query.",
            },
            goto="queries_generator",
        )

    def _searcher_node(self, state: LiteratureReviewerState) -> Dict[str, Any]:
        """
        Deterministic step by step papers retrieval approach.

        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        main_query = state["main_query"]
        queries = state["queries"]

        candidates = self._discover(main_query, queries)
        selected = self._relevance_gate(candidates=candidates, goal=state["research_goal"], hypothesis=state["hypothesis"])
        self._ingest_selected(selected)
        evidence = self._retrieve(main_query, queries)

        merged = self._merge_evidence(state.get("raw_data") or [], evidence)
        return {"raw_data": merged}

    def _merge_evidence(self, existing: List[EvidenceItem], new: List[EvidenceItem]) -> List[EvidenceItem]:
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

    def _discover(self, main_query: str, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Search arXiv + Semantic Scholar; merge + dedup ingestable candidates.

        Args:
            main_query (str): Main hypothesis query.
            queries (str): List of query variants.
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing informatin about the retrieve papers.
        """
        terms = [main_query] + queries[:DISCOVERY_VARIANTS]
        candidates: Dict[tuple, Dict[str, Any]] = {}

        for term in terms:
            for tool, kwargs in (
                (search_arxiv, {"keyword": term, "max_results": 5}),
                (search_schoolar, {"query": term, "max_results": 5}),
            ):
                try:
                    results = tool(**kwargs)
                except Exception:
                    continue
                for r in results:
                    key = (r.get("source"), r.get("external_id"))

                    if key not in candidates and r.get("pdf_url") and r.get("external_id"):
                        candidates[key] = r

        return list(candidates.values())

    def _relevance_gate(self, candidates: List[Dict[str, Any]], goal: str, hypothesis: str) -> List[Dict[str, Any]]:
        """
        LLM picks which candidates are worth ingesting.

        Args:
            candidates (List[Dict[str, Any]]): List of retrieve papers to evaluate.
            goal (str): Research general goal.
            hypothesis (str): Hypothesis being analyze.
        Returns:
            List[Dict[str, Any]]: List of worth ingesting papers.
        """
        if not candidates:
            return []

        try:
            enumerated = [
                {"index": i, "title": c["title"], "abstract": c["summary"]}
                for i, c in enumerate(candidates)]
            
            result = self.llm.invoke(
                prompt=RELEVANCE_GATE_PROMPT,
                input={"research_goal": goal, "hypothesis": hypothesis, "candidates": enumerated},
                output_schema=RelevanceGateOutput,
            )
            data = result if isinstance(result, RelevanceGateOutput) else RelevanceGateOutput(**result.model_dump())
            indices = data.selected_indices

        except Exception:
            indices = list(range(len(candidates)))

        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        return selected[:INGEST_CAP]

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
            try:
                if self.vector_store.paper_exists(source, external_id):
                    continue
                paper_meta = {
                    "title": c["title"],
                    "source": source,
                    "external_id": external_id,
                    "url": c.get("abstract_url"),
                    "abstract": c.get("summary") or None,
                    "authors": c.get("authors") or [],
                }
                self.ingestion.run(url=c["pdf_url"], paper_meta=paper_meta)
            except Exception:
                continue

    def _retrieve(self, main_query: str, queries: List[str]) -> List[EvidenceItem]:
        """
        Hybrid-retrieve full-text chunks and normalize into EvidenceItems.
        
        Args:
            main_query (str): Main hypothesis query.
            queries (str): List of query variants.
        Returns:
            List[EvidenceItem]: List of formatted evidence objects.
        """
        try:
            results = self.vector_store.retrieve_documents(main_query=main_query, queries=queries)
        except Exception:
            return []

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

    def _data_evaluator_node(self, state: LiteratureReviewerState) -> Dict[str, Any]:
        """
        Judge whether the gathered evidence is sufficient; produce feedback.
        
        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=DATA_VALIDATION_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypothesis": state["hypothesis"],
                    "raw_data": [e.model_dump() for e in state["raw_data"]],
                },
                output_schema=DataValidatorOutput,
            )
            data = result if isinstance(result, DataValidatorOutput) else DataValidatorOutput(**result.model_dump())
            return {"has_sufficient_data": data.validation_label, "queries_feedback": data.feedback}
        
        except Exception:
            return {
                "has_sufficient_data": "insufficient",
                "queries_feedback": "Something went wrong when evaluating the data.",
            }

    def _data_router(self, state: LiteratureReviewerState) -> Command[Literal["finding_extractor", "queries_generator"]]:
        """
        Extract findings if sufficient; else loop for more data (capped).
        
        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            Command: Langgraph function that redirects to specified node in the graph.
        """
        if state["has_sufficient_data"] == "sufficient":
            return Command(goto="finding_extractor")

        # Extract from whatever we gathered.
        if state["search_retries"] + 1 >= MAX_RETRIES_PER_NODE:
            return Command(goto="finding_extractor")

        return Command(
            update={"search_retries": state["search_retries"] + 1},
            goto="queries_generator",
        )

    def _finding_extractor_node(self, state: LiteratureReviewerState) -> Dict[str, Any]:
        """
        Turn the gathered evidence into structured findings.
        
        Args:
            state (LiteratureReviewerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=FINDING_EXTRATOR_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypothesis": state["hypothesis"],
                    "raw_data": [e.model_dump() for e in state["raw_data"]],
                },
                output_schema=FindingExtractorOutput,
            )
            data = result if isinstance(result, FindingExtractorOutput) else FindingExtractorOutput(**result.model_dump())
            return {"findings": data.findings}
        
        except Exception:
            return {"findings": []}

    def run(self, research_goal: str, hypothesis: str) -> List[Finding]:
        """
        Run the literature reviewer and return the extracted findings.
        
        Args:
            research_goal (str): Main research goal.
            hypothesis (str): Hypothesis being analyze.
        Returns:
            List[Finding]: List of formatted and analyze finding, extracted from the retrieved papers.
        """
        initial_state = LiteratureReviewerState(
            research_goal=research_goal,
            hypothesis=hypothesis,
            main_query="",
            queries=[],
            queries_feedback="",
            raw_data=[],
            has_sufficient_data="insufficient",
            findings=[],
            query_gen_retries=0,
            search_retries=0,
        )
        result = self.graph.invoke(initial_state)
        return result["findings"]