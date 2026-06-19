from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph

from src.researcher.models import Hypothesis
from src.researcher.agents.literature_reviewer.utils.models import Finding
from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from .utils.prompts import (
    GENERATE_SUMMARY_PROMPT,
    LITERATURE_ANALYSIS_PROMPT,
    GENERATE_REPORT_PROMPT
)
from .utils.models import (
    SourceMeta,
    Citation,
    SummaryOutput,
    LiteratureAnalysisOutput,
    ReportOutput
)

class WritterState(TypedDict):
    research_goal: str
    hypotheses: List[Hypothesis]
    findings: List[Finding]

    sources: List[SourceMeta]
    citations: List[Citation]
    sources_block: str

    summary: str
    literature_review: LiteratureAnalysisOutput | None
    report: ReportOutput | None

class Writter:
    """
    Writing agent. Given the research goal, hypotheses, and extracted findings,
    it produces an executive summary, a themed synthesis of the literature, and a
    final structured research report (rendered to markdown for the human).
    """
    def __init__(self):
        """
        Initializes the writter agent.
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature
        ))
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Builds and compiles the agent graph.

        Returns:
            StateGraph: The compiled LangGraph graph.
        """
        graph = StateGraph(WritterState)

        graph.add_node("literature_reviewer", self._literature_reviewer_node)
        graph.add_node("report_generator", self._research_report_generator_node)
        graph.add_node("summary_generator", self._research_summary_generator_node)

        graph.set_entry_point("literature_reviewer")
        graph.add_edge("literature_reviewer", "report_generator")
        graph.add_edge("report_generator", "summary_generator")
        graph.set_finish_point("summary_generator")

        return graph.compile()

    def _literature_reviewer_node(self, state: WritterState) -> Dict[str, Any]:
        """
        Synthesizes the extracted findings into themes, consensus, contradictions
        and gaps. Consumed by the report node as the literature substrate.

        Args:
            state (WritterState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            result = self.llm.invoke(
                prompt=LITERATURE_ANALYSIS_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "findings": [f.model_dump() for f in state["findings"]],
                    "sources_block": state["sources_block"],
                },
                output_schema=LiteratureAnalysisOutput,
            )
            data = result if isinstance(result, LiteratureAnalysisOutput) else LiteratureAnalysisOutput(**result.model_dump())
            return {"literature_review": data}

        except Exception:
            return {"literature_review": None}

    def _research_report_generator_node(self, state: WritterState) -> Dict[str, Any]:
        """
        Generates a final structured report detailing what was explored, what was
        found, and what remains open, drawing on the literature synthesis.

        Args:
            state (WritterState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        try:
            literature_review = state.get("literature_review")
            result = self.llm.invoke(
                prompt=GENERATE_REPORT_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "hypotheses": [h.model_dump() for h in state["hypotheses"]],
                    "literature_review": literature_review.model_dump() if literature_review else {},
                    "sources_block": state["sources_block"],
                },
                output_schema=ReportOutput,
            )
            data = result if isinstance(result, ReportOutput) else ReportOutput(**result.model_dump())
            # References are built deterministically, never by the model.
            data.references = state["citations"]
            return {"report": data}

        except Exception:
            return {"report": None}

    def _research_summary_generator_node(self, state: WritterState) -> Dict[str, Any]:
        """
        Generates an executive abstract of the finished report.

        Args:
            state (WritterState): Graph state.
        Returns:
            dict[str, any]: Dictionary containing the properties to update in the global state.
        """
        report = state.get("report")
        if report is None:
            return {"summary": ""}

        try:
            result = self.llm.invoke(
                prompt=GENERATE_SUMMARY_PROMPT,
                input={
                    "research_goal": state["research_goal"],
                    "report": report.model_dump(),
                },
                output_schema=SummaryOutput,
            )
            data = result if isinstance(result, SummaryOutput) else SummaryOutput(**result.model_dump())
            return {"summary": data.summary}

        except Exception:
            return {"summary": ""}

    @staticmethod
    def _build_citations(
        findings: List[Finding],
        sources: List[SourceMeta],
    ) -> List[Citation]:
        """
        Assigns citation numbers deterministically by first appearance of each
        paper_id across the findings, looking up title/url from the provided
        sources. Only papers actually cited (present in findings) are included.

        Args:
            findings (List[Finding]): The findings whose source_paper_ids drive citation order.
            sources (List[SourceMeta]): Paper metadata supplied by the orchestrator.
        Returns:
            List[Citation]: Ordered, numbered citations.
        """
        meta_by_id = {s.paper_id: s for s in sources}

        citations: List[Citation] = []
        seen: Dict[str, int] = {}
        for finding in findings:
            paper_id = finding.source_paper_id
            if not paper_id or paper_id in seen:
                continue

            number = len(citations) + 1
            seen[paper_id] = number
            meta = meta_by_id.get(paper_id)
            citations.append(Citation(
                number=number,
                paper_id=paper_id,
                title=meta.title if meta else None,
                url=meta.url if meta else None,
            ))

        return citations

    @staticmethod
    def _format_sources_block(citations: List[Citation]) -> str:
        """
        Renders the numbered citations into the [n] reference block shown to the
        model so its inline markers match the deterministic numbering.

        Args:
            citations (List[Citation]): The numbered citations.
        Returns:
            str: A newline-separated "[n] title (url)" block, or a placeholder if empty.
        """
        if not citations:
            return "(no sources available)"

        lines = []
        for c in citations:
            title = c.title or c.paper_id
            suffix = f" ({c.url})" if c.url else ""
            lines.append(f"[{c.number}] {title}{suffix}")

        return "\n".join(lines)

    def run(self,
            research_goal: str,
            hypotheses: List[Hypothesis],
            findings: List[Finding],
            sources: List[SourceMeta] | None = None) -> ReportOutput | None:
        """
        Runs the Writter agent.

        Args:
            research_goal (str): The research goal being written up.
            hypotheses (List[Hypothesis]): The hypotheses produced during the session.
            findings (List[Finding]): The claims extracted from the literature.
            sources (List[SourceMeta] | None): Paper metadata (title/url) for citations.
        Returns:
            ReportOutput | None: The structured report, or None if generation failed.
        """
        sources = sources or []
        citations = self._build_citations(findings, sources)
        sources_block = self._format_sources_block(citations)

        initial_state = WritterState(
            research_goal=research_goal,
            hypotheses=hypotheses,
            findings=findings,
            sources=sources,
            citations=citations,
            sources_block=sources_block,
            summary="",
            literature_review=None,
            report=None,
        )

        results = self.graph.invoke(initial_state)
        return results.get("report")