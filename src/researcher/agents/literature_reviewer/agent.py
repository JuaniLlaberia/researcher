from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.core.logging import get_logger
from src.researcher.tools.searcher import Searcher
from src.researcher.models import EvidenceItem
from .utils.models import (
    Finding,
    DataValidatorOutput,
    QueriesOutput,
    FindingExtractorOutput,
    RelevanceGateOutput,
)
from .utils.prompts import (
    QUERIES_GENERATION_PROMPT,
    DATA_VALIDATION_PROMPT,
    FINDING_EXTRATOR_PROMPT,
    RELEVANCE_GATE_PROMPT,
)

MAX_RETRIES_PER_NODE = 3
INGEST_CAP = 2

log = get_logger("literature_reviewer")

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
    def __init__(self, searcher: Searcher | None = None) -> None:
        """
        Initializes the LiteratureReviwer class.

        Args:
            searcher (Searcher | None): Shared searcher to reuse (avoids loading a
                second Docling pipeline / vector store). Self-constructs if omitted.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        ))

        self.searcher = searcher or Searcher()

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
            log.info("queries: main=%r + %d variants", data.main_query, len(data.queries))
            return {"main_query": data.main_query, "queries": data.queries}

        except Exception:
            log.exception("query generation failed")
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

        gate = lambda candidates: self._relevance_gate(
            candidates=candidates, goal=state["research_goal"], hypothesis=state["hypothesis"]
        )
        evidence = self.searcher.gather(main_query, queries, gate=gate)

        merged = self.searcher.merge_evidence(state.get("raw_data") or [], evidence)
        return {"raw_data": merged}

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
            log.exception("relevance gate failed — passing all %d candidates", len(candidates))
            indices = list(range(len(candidates)))

        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        return selected[:INGEST_CAP]

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
            log.info("data sufficiency: %s", data.validation_label)
            return {"has_sufficient_data": data.validation_label, "queries_feedback": data.feedback}

        except Exception:
            log.exception("data evaluation failed — treating as insufficient")
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
            log.info("extracted %d findings from %d evidence chunks", len(data.findings), len(state["raw_data"]))
            return {"findings": data.findings}

        except Exception:
            log.exception("finding extraction failed")
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