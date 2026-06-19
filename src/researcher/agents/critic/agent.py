from typing import TypedDict, Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.researcher.tools.searcher import Searcher
from src.researcher.models import EvidenceItem

from .utils.models import (AdversarialQueriesOutput,
                           AdversarialRelevanceGateOutput,
                           CriticAssessment)
from .utils.prompts import (ADVERSARIAL_QUERIES_GENERATION_PROMPT,
                            ADVERSARIAL_RELEVANCE_GATE_PROMPT,
                            HYPOTHESIS_ASSESSMENT_PROMPT)

MAX_RETRIES_PER_NODE = 3
INGEST_CAP = 3

class CriticState(TypedDict):
    research_goal: str
    hypothesis: str

    queries: List[str]
    raw_data: List[EvidenceItem]

    assessment: CriticAssessment | None

    query_gen_retries: int

class Critic:
    """
    Critic agent that generates an analysis and verdict of the hypothesis.
    """
    def __init__(self):
        """
        Initializes Critic pipeline class.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ))
        self.searcher = Searcher()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Builds and compiles the agent graph.

        Returns:
            StateGraph: The compiled LangGraph graph.
        """
        graph = StateGraph(CriticState)

        graph.add_node("adv_queries_generator", self._adversarial_queries_generation)
        graph.add_node("queries_validator", self._queries_validator)
        graph.add_node("literature_searcher", self._literature_search_node)
        graph.add_node("hypothesis_assessor", self._hypothesis_assessment_node)

        graph.set_entry_point("adv_queries_generator")
        graph.add_edge("adv_queries_generator", "queries_validator")
        graph.add_edge("literature_searcher", "hypothesis_assessor")
        graph.set_finish_point("hypothesis_assessor")

        return graph.compile()

    def _adversarial_queries_generation(self, state: CriticState) -> Dict[str, Any]:
        """
        Generates adversarial (oposite) queries based on the hypothesis.

        Args:
            state (CriticState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=ADVERSARIAL_QUERIES_GENERATION_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypothesis": state["hypothesis"],
                },
                output_schema=AdversarialQueriesOutput,
            )
            data = result if isinstance(result, AdversarialQueriesOutput) else AdversarialQueriesOutput(**result.model_dump())
            return {"queries": data.queries}
        
        except Exception:
            return {"queries": []}

    def _queries_validator(self, state: CriticState) -> Command[Literal["literature_searcher", "adv_queries_generator"]]:
        """
        Route to the critic if queries are valid, else regenerate (capped).

        Args:
            state (CriticState): Graph state.
        Returns:
            Command: Langgraph function that redirects to specified node in the graph.
        """
        if len(state["queries"]) >= 1:
            return Command(update={"query_gen_retries": 0}, goto="literature_searcher")

        if state["query_gen_retries"] + 1 >= MAX_RETRIES_PER_NODE:
            return Command(goto=END)

        return Command(
            update={"query_gen_retries": state["query_gen_retries"] + 1},
            goto="adv_queries_generator",
        )

    def _literature_search_node(self, state: CriticState) -> Dict[str, Any]:
        """
        Searches for evidence that contradicts the hypothesis.
        
        Args:
            state (CriticState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        main_query = state["queries"][0]
        queries = state["queries"]

        gate = lambda candidates: self._relevance_gate(
            candidates=candidates, goal=state["research_goal"], hypothesis=state["hypothesis"]
        )
        evidence = self.searcher.gather(main_query=main_query, queries=queries, gate=gate)

        merged = self.searcher.merge_evidence(state.get("raw_data") or [], evidence)
        return {"raw_data": merged}

    def _relevance_gate(self, candidates: List[Dict[str, Any]], goal: str, hypothesis: str) -> List[Dict[str, Any]]:
        """
        LLM picks which candidates are worth ingesting (Adversarial, trying to contradict the hypothesis).

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
                prompt=ADVERSARIAL_RELEVANCE_GATE_PROMPT,
                input={"research_goal": goal, "hypothesis": hypothesis, "candidates": enumerated},
                output_schema=AdversarialRelevanceGateOutput,
            )
            data = result if isinstance(result, AdversarialRelevanceGateOutput) else AdversarialRelevanceGateOutput(**result.model_dump())
            indices = data.selected_indices

        except Exception:
            indices = list(range(len(candidates)))

        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        return selected[:INGEST_CAP]

    def _hypothesis_assessment_node(self, state: CriticState) -> Dict[str, Any]:
        """
        Analyze the hypothesis against the gathered (adversarial) evidence and
        render a verdict in a single pass: contradictions, hidden assumptions,
        falsifiability, rationale, and the verdict.

        Args:
            state (CriticState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=HYPOTHESIS_ASSESSMENT_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypothesis": state["hypothesis"],
                    "raw_data": [e.model_dump() for e in state["raw_data"]],
                },
                output_schema=CriticAssessment,
            )
            data = result if isinstance(result, CriticAssessment) else CriticAssessment(**result.model_dump())
            return {"assessment": data}

        except Exception:
            return {"assessment": None}

    def run(self, research_goal: str, hypothesis: str) -> CriticAssessment:
        """
        Runs Critic pipeline.

        Args:
            research_goal (str): Main research goal.
            hypothesis (str): Hypothesis being analyze.
        Returns:
            CriticAssessment: Analysis and veredict of the hypothesis.
        """
        initial_state = CriticState(
            research_goal=research_goal,
            hypothesis=hypothesis,
            queries=[],
            raw_data=[],
            assessment=None,
            query_gen_retries=0,
        )

        results = self.graph.invoke(initial_state)
        assessment = results.get("assessment")

        if assessment is None:
            assessment = CriticAssessment(
                contradictions=[],
                assumptions=[],
                falsifiability_score=0.0,
                rationale="The critic could not gather enough evidence to assess the hypothesis.",
                verdict="needs_more_evidence",
            )

        return assessment