from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.core.logging import get_logger
from src.researcher.models import EvidenceItem
from src.researcher.tools.searcher import Searcher
from .utils.models import (
    GeneratedHypothesis,
    GenerationOutput,
    HypothesisScore,
    ScoringOutput,
    ScoredHypothesis,
    RefineOutput,
)
from .utils.prompts import (
    HYPOTHESES_GENERATION_PROMPT,
    SCORE_HYPOTHESES_PROMPT,
    REFINE_HYPOTHESIS_PROMPT,
)

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
TOP_K = 3
# Re-scope the goal if fewer than this many chunks retrieved
THINNESS_THRESHOLD = 3

log = get_logger("hypothesizer")

class HypothesizerState(TypedDict):
    research_goal: str
    feedback: str

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
    def __init__(self, searcher: Searcher | None = None):
        """
        Initializes the Hypothesizer pipeline class.

        Args:
            searcher (Searcher | None): Shared searcher to reuse (avoids loading a
                second Docling pipeline / vector store). Self-constructs if omitted.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ))
        self.searcher = searcher or Searcher()
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

    def _base_knowledge_retriever_node(self, state: HypothesizerState) -> Dict[str, Any]:
        """
        Retrieve grounding context for the goal from the vector store. If the
        knowledge base is thin, lazily scope the goal (search + ingest a few
        papers, no relevance gate) and retrieve again. If still empty, generation
        falls back to the LLM's prior knowledge.

        Args:
            state (HypothesizerState): Graph state.
        Returns:
            dict[str, Any]: State update with the gathered evidence.
        """
        goal = state["research_goal"]
        evidence = self.searcher.retrieve(goal, [goal])

        if len(evidence) < THINNESS_THRESHOLD:
            log.info("knowledge base thin (%d chunks) — scoping the goal via search", len(evidence))
            try:
                evidence = self.searcher.gather(goal, [goal], gate=None)
            except Exception:
                log.exception("goal scoping failed — falling back to LLM priors")

        log.info("base knowledge: %d evidence chunks", len(evidence))
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
                    "feedback": state.get("feedback") or "",
                },
                output_schema=GenerationOutput,
            )
            data = result if isinstance(result, GenerationOutput) else GenerationOutput(**result.model_dump())
            log.info("generated %d raw hypotheses", len(data.hypotheses[:MAX_HYPOTHESES]))
            return {"raw_hypotheses": data.hypotheses[:MAX_HYPOTHESES]}

        except Exception:
            log.exception("hypothesis generation failed")
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
            log.info("scored %d hypotheses", len(data.scores))
            return {"scored_hypotheses": data.scores}

        except Exception:
            log.exception("hypothesis scoring failed")
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

        kept = survivors[:TOP_K]
        log.info("filtered: %d/%d above threshold %.2f, keeping top %d",
                 len(survivors), len(scored), FILTER_THRESHOLD, len(kept))
        return {"hypotheses": kept}

    def run(self, research_goal: str, feedback: str = "") -> List[ScoredHypothesis]:
        """
        Runs the hypothesizer pipeline, where we generate and evaluate hypotheses based on research goal.

        Args:
            research_goal (str): Goal being research it.
            feedback (str): Optional human steering for (re)generation; "" when absent.
        Returns:
            List[ScoredHypothesis]: The filtered, ranked hypotheses.
        """
        initial_state = HypothesizerState(
            research_goal=research_goal,
            feedback=feedback,
            raw_data=[],
            raw_hypotheses=[],
            scored_hypotheses=[],
            hypotheses=[]
        )

        results = self.graph.invoke(initial_state)
        return results["hypotheses"]

    def refine(self, research_goal: str, hypothesis: str, feedback: str) -> ScoredHypothesis:
        """
        Refine a single hypothesis in response to a critique, and re-score it on the
        same rubric used at generation so the child's confidence is honest (not the
        parent's stale score). One LLM call; no full graph.

        Args:
            research_goal (str): The research goal.
            hypothesis (str): The original hypothesis text being refined.
            feedback (str): Condensed critic feedback to address.
        Returns:
            ScoredHypothesis: The refined hypothesis with its composite score.
        """
        result = self.llm.invoke(
            prompt=REFINE_HYPOTHESIS_PROMPT,
            input={
                "research_goal": research_goal,
                "hypothesis": hypothesis,
                "feedback": feedback,
            },
            output_schema=RefineOutput,
        )
        data = result if isinstance(result, RefineOutput) else RefineOutput(**result.model_dump())

        composite = sum(getattr(data, dim) * weight for dim, weight in SCORE_WEIGHTS.items())
        return ScoredHypothesis(
            text=data.text,
            score=composite,
            relevance=data.relevance,
            testability=data.testability,
            specificity=data.specificity,
            plausibility=data.plausibility,
            novelty=data.novelty,
            rationale=data.rationale,
        )
