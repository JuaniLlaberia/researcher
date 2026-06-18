from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.researcher.agents.literature_reviewer.utils.models import EvidenceItem
from src.researcher.tools.scope_goal import scope_goal
from src.db.vector_store import VectorStore
from .utils.models import (
    GeneratedHypothesis,
    GenerationOutput,
    HypothesisScore,
    ScoringOutput,
    ScoredHypothesis,
)
from .utils.prompts import HYPOTHESES_GENERATION_PROMPT, SCORE_HYPOTHESES_PROMPT

MIN_HYPOTHESES = 4
MAX_HYPOTHESES = 8

SCORE_WEIGHTS = {
    "testability": 0.30,
    "relevance": 0.30,
    "specificity": 0.20,
    "plausibility": 0.15,
    "novelty": 0.05,
}

FILTER_THRESHOLD = 0.5
TOP_K = 5
# Re-scope the goal if fewer than this many chunks retrieved
THINNESS_THRESHOLD = 3      

class HypothesizerState(TypedDict):
    research_goal: str

    raw_data: List[EvidenceItem]

    raw_hypotheses: List[GeneratedHypothesis]
    scored_hypotheses: List[HypothesisScore]
    hypotheses: List[ScoredHypothesis]

class Hypothesizer:
    """
    Hypothesis engine. Given a research goal it retrieves grounding context,
    generates a batch of candidate hypotheses, scores them on a rubric, and
    returns the filtered, ranked survivors as ScoredHypothesis objects.
    """
    def __init__(self):
        """
        Initializes the Hypothesizer pipeline class.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ))
        self.vector_store = VectorStore()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Builds and compiles the agent graph.

        Returns:
            StateGraph: The compiled LangGraph graph.
        """
        graph = StateGraph(HypothesizerState)

        graph.add_node("bk_retriever_node", self._base_knowledge_retriever_node)
        graph.add_node("hypotheses_generator", self._hypotheses_generator_node)
        graph.add_node("hypotheses_scorer", self._hypotheses_scorer_node)
        graph.add_node("hypotheses_filter", self._hypotheses_filter_node)

        graph.set_entry_point("bk_retriever_node")
        graph.add_edge("bk_retriever_node", "hypotheses_generator")
        graph.add_edge("hypotheses_generator", "hypotheses_scorer")
        graph.add_edge("hypotheses_scorer", "hypotheses_filter")
        graph.set_finish_point("hypotheses_filter")

        return graph.compile()

    def _retrieve_evidence(self, research_goal: str) -> List[EvidenceItem]:
        """
        Hybrid-retrieve grounding chunks for the goal and normalize them into
        EvidenceItems. The goal is used as its own semantic query so the vector
        (semantic) arm runs, not just keyword search.

        Args:
            research_goal (str): The goal to retrieve context for.
        Returns:
            List[EvidenceItem]: Retrieved evidence (empty on failure).
        """
        try:
            results = self.vector_store.retrieve_documents(
                main_query=research_goal, queries=[research_goal]
            )
        except Exception as e:
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

    def _base_knowledge_retriever_node(self, state: HypothesizerState) -> Dict[str, Any]:
        """
        Retrieve grounding context for the goal from the vector store. If the
        knowledge base is thin, lazily scope the goal (search + ingest a few
        papers) and retrieve again. If still empty, generation falls back to the
        LLM's prior knowledge.

        Args:
            state (HypothesizerState): Graph state.
        Returns:
            dict[str, Any]: State update with the gathered evidence.
        """
        goal = state["research_goal"]
        evidence = self._retrieve_evidence(goal)

        if len(evidence) < THINNESS_THRESHOLD:
            try:
                print("SCOPING GOAL")
                scope_goal(goal)
                evidence = self._retrieve_evidence(goal)
            except Exception as e:
                print(e)

        print("EVIDENCE", evidence)
        return {"raw_data": evidence}

    def _hypotheses_generator_node(self, state: HypothesizerState) -> Dict[str, Any]:
        """
        Generates N hypothesis based on the research goal and the knowledge base (optional).

        Args:
            state (HypothesizerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=HYPOTHESES_GENERATION_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "raw_data": [e.model_dump() for e in state["raw_data"]],
                },
                output_schema=GenerationOutput,
            )
            data = result if isinstance(result, GenerationOutput) else GenerationOutput(**result.model_dump())
            print("H1", data)
            return {"raw_hypotheses": data.hypotheses[:MAX_HYPOTHESES]}

        except Exception:
            return {"raw_hypotheses": []}

    def _hypotheses_scorer_node(self, state: HypothesizerState) -> Dict[str, Any]:
        """
        Generates scores for each hypothesis based on: relevance, testability, specificity, plausibility and novelty.

        Args:
            state (HypothesizerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        enumerated = [
            {"index": i, "text": h.text} for i, h in enumerate(state["raw_hypotheses"])
        ]
        if not enumerated:
            return {"scored_hypotheses": []}

        try:
            result = self.llm.invoke(
                prompt=SCORE_HYPOTHESES_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "raw_data": [e.model_dump() for e in state["raw_data"]],
                    "hypotheses": enumerated,
                },
                output_schema=ScoringOutput,
            )
            data = result if isinstance(result, ScoringOutput) else ScoringOutput(**result.model_dump())
            print("H2", data)
            return {"scored_hypotheses": data.scores}

        except Exception:
            return {"scored_hypotheses": []}

    def _hypotheses_filter_node(self, state: HypothesizerState) -> Dict[str, Any]:
        """
        Join each score back to its hypothesis text (by index), compute the
        weighted composite, then filter by threshold and keep the top-K.

        Args:
            state (HypothesizerState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        raw_hypotheses = state["raw_hypotheses"]
        scores_by_index = {s.index: s for s in state["scored_hypotheses"]}

        scored: List[ScoredHypothesis] = []
        for i, hypothesis in enumerate(raw_hypotheses):
            score = scores_by_index.get(i)
            if score is None:
                continue

            composite = sum(getattr(score, dim) * weight for dim, weight in SCORE_WEIGHTS.items())

            scored.append(ScoredHypothesis(
                text=hypothesis.text,
                score=composite,
                relevance=score.relevance,
                testability=score.testability,
                specificity=score.specificity,
                plausibility=score.plausibility,
                novelty=score.novelty,
                rationale=score.rationale,
            ))

        survivors = [h for h in scored if h.score >= FILTER_THRESHOLD]
        survivors.sort(key=lambda h: h.score, reverse=True)

        return {"hypotheses": survivors[:TOP_K]}

    def run(self, research_goal: str) -> List[ScoredHypothesis]:
        """
        Runs the hypothesizer pipeline, where we generate and evaluate hypotheses based on research goal.

        Args:
            research_goal (str): Goal being research it.
        Returns:
            List[ScoredHypothesis]: The filtered, ranked hypotheses.
        """
        initial_state = HypothesizerState(
            research_goal=research_goal,
            raw_data=[],
            raw_hypotheses=[],
            scored_hypotheses=[],
            hypotheses=[]
        )

        results = self.graph.invoke(initial_state)
        return results["hypotheses"]
