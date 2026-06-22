from uuid import uuid4
from typing import Literal, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command, Send, interrupt
from langchain_core.messages import HumanMessage, AIMessage
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from src.core.llm import LLM, LLMConfig
from src.core.config import settings
from src.core.logging import get_logger, log_stage
from src.db.memory import generate_summary_memory
from src.researcher.tools.searcher import Searcher
from src.researcher.state import ResearchState, HumanInTheLoopEnum, RunTypeEnum
from src.researcher.models import Hypothesis
from src.researcher.utils.models import IntentEnum, IntentClassificationOutput, RespondOutput
from src.researcher.utils.prompts import CLASSIFY_INTENT_PROMPT, RESPOND_PROMPT
from src.researcher.adapters import scored_to_hypothesis, apply_critic, stamp_findings, park_parent
from src.researcher.agents.hypothesizer.agent import Hypothesizer
from src.researcher.agents.literature_reviewer.agent import LiteratureReviewer
from src.researcher.agents.critic.agent import Critic
from src.researcher.agents.writer.agent import Writer
from src.researcher.agents.writer.utils.models import SourceMeta, ReportOutput
from src.db.session_store import (
    create_research,
    start_session_in_research,
    load_state,
    save_state,
    load_paper_sources,
    set_pending_thread,
)

NODE_CLASSIFIER = "intent_classifier"
NODE_INTENT_ROUTER = "intent_router"
NODE_HYPOTHESIZER = "hypothesizer"
NODE_HYPOTHESIS_REFINER = "hypothesis_refiner"
NODE_HYPOTHESIZER_ROUTER = "hypothesizer_router"
NODE_LITERATURE = "literature_reviewer"
NODE_CRITIC = "critic"
NODE_CRITIC_JOIN = "critic_join"
NODE_WRITER = "writer"
NODE_GATE_HYPOTHESES = "gate_hypotheses"
NODE_GATE_REPORT = "gate_report"
NODE_RESPONDER = "responder"
NODE_SUMMARY_ROUTER = "summary_router"
NODE_SUMMARIZER = "summarization"
NODE_PERSIST = "persist"

INTENT_MAP = {
    IntentEnum.GENERATE_HYPOTHESIS: NODE_HYPOTHESIZER,
    IntentEnum.REVIEW_LITERATURE: NODE_LITERATURE,
    IntentEnum.CRITIQUE_HYPOTHESIS: NODE_CRITIC,
    IntentEnum.RESEARCH_REPORT: NODE_WRITER,
    IntentEnum.CHAT: NODE_RESPONDER,
}
FANOUT_NODES = {NODE_LITERATURE, NODE_CRITIC}
SUMMARY_EVERY = 12
CONTEXT_MESSAGES = 10
REFINE_CAP = 2

GATE_HYPOTHESES = "hypotheses"
GATE_REFINE = "refine"
GATE_REPORT = "report"

ARMED_GATES: dict[HumanInTheLoopEnum, set[str]] = {
    HumanInTheLoopEnum.AUTONOMOUS: set(),
    HumanInTheLoopEnum.CHECKPOINT: {GATE_HYPOTHESES, GATE_REPORT},
    HumanInTheLoopEnum.INTERACTIVE: {GATE_HYPOTHESES, GATE_REFINE, GATE_REPORT},
}

log = get_logger("orchestrator")

def _short(hypothesis: Hypothesis) -> str:
    """Short, log-friendly handle for a hypothesis (id prefix + truncated text)."""
    return f"{hypothesis.id[:8]} {hypothesis.text[:60]!r}"

class Orquestrator:
    """
    Assistant supervisor. Classifies the per-turn intent, routes to the relevant
    specialist (with per-hypothesis fan-out for the literature/critique work),
    funnels every route through a single responder, then persists the turn.
    """
    def __init__(self, checkpointer):
        """
        Initializes the orchestrator: shared LLM, hoisted agents, compiled graph.

        Use `await Orquestrator.create()` for normal (Postgres-backed) construction;
        the checkpointer is injected so tests can pass an in-memory saver.

        Args:
            checkpointer: A LangGraph checkpointer used to persist in-flight graph
                state (required for HITL interrupt/resume).
        """
        self.llm = LLM(config=LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        ))
        self.searcher = Searcher()

        self.hypothesizer = Hypothesizer(searcher=self.searcher)
        self.literature_reviewer = LiteratureReviewer(searcher=self.searcher)
        self.critic = Critic(searcher=self.searcher)
        self.writer = Writer()

        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    @classmethod
    async def create(cls) -> "Orquestrator":
        """
        Async factory: open a Postgres-backed checkpointer (creating its tables if
        needed) and build the orchestrator. The connection pool stays open for the
        orchestrator's lifetime.

        Returns:
            Orquestrator: A ready, HITL-capable orchestrator.
        """
        pool = AsyncConnectionPool(
            conninfo=settings.database_dsn,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        log.info("checkpointer ready (postgres)")

        self = cls(checkpointer=saver)
        self._pool = pool
        return self

    def _build_graph(self):
        """
        Wires the supervisor graph: classifier -> router -> {agents | responder},
        with every route converging on the responder, then summary -> persist -> END.

        Returns:
            CompiledStateGraph: The compiled orchestrator graph.
        """
        graph = StateGraph(ResearchState)

        graph.add_node(NODE_CLASSIFIER, self._intent_classifier)
        graph.add_node(NODE_INTENT_ROUTER, self._intent_router)
        graph.add_node(NODE_HYPOTHESIZER, self._hypothesizer_node)
        graph.add_node(NODE_HYPOTHESIS_REFINER, self._hypothesis_refiner_node)
        graph.add_node(NODE_HYPOTHESIZER_ROUTER, self._hypothesizer_router)
        graph.add_node(NODE_LITERATURE, self._literature_reviewer_node)
        graph.add_node(NODE_CRITIC, self._critic_node)
        graph.add_node(NODE_CRITIC_JOIN, self._critic_join)
        graph.add_node(NODE_GATE_HYPOTHESES, self._gate_hypotheses_node)
        graph.add_node(NODE_GATE_REPORT, self._gate_report_node)
        graph.add_node(NODE_WRITER, self._writer_node)
        graph.add_node(NODE_RESPONDER, self._responder_node)
        graph.add_node(NODE_SUMMARY_ROUTER, self._summary_router)
        graph.add_node(NODE_SUMMARIZER, self._summarization_node)
        graph.add_node(NODE_PERSIST, self._persist_node)

        graph.add_edge(START, NODE_CLASSIFIER)
        graph.add_edge(NODE_CLASSIFIER, NODE_INTENT_ROUTER)
        graph.add_edge(NODE_HYPOTHESIZER, NODE_GATE_HYPOTHESES)
        graph.add_edge(NODE_RESPONDER, NODE_SUMMARY_ROUTER)
        graph.add_edge(NODE_SUMMARIZER, NODE_PERSIST)
        graph.add_edge(NODE_PERSIST, END)

        return graph.compile(checkpointer=self.checkpointer)

    def _branch_payload(self, state: ResearchState, hypothesis: Hypothesis, pipeline: bool) -> Dict[str, Any]:
        """
        Build the per-hypothesis state a Send branch receives. A branch only sees this
        payload (not the merged graph state), so it carries everything it needs: the
        goal, the hypothesis, and whether to keep flowing down the pipeline.

        Args:
            state (ResearchState): Graph state.
            hypothesis (Hypothesis): The hypothesis this branch handles.
            pipeline (bool): Whether the branch continues the full pipeline or stops after its step.
        Returns:
            dict[str, Any]: The Send branch payload.
        """
        return {
            "goal": state["goal"],
            "current_hypothesis": hypothesis,
            "pipeline": pipeline,
        }

    def _refine_payload(self, state: ResearchState, hypothesis: Hypothesis) -> Dict[str, Any]:
        """
        Build the payload for a refine branch: the parent hypothesis plus the critic
        feedback it must address.

        Args:
            state (ResearchState): Graph state.
            hypothesis (Hypothesis): The parent hypothesis to refine.
        Returns:
            dict[str, Any]: The Send branch payload for the refiner.
        """
        return {
            "goal": state["goal"],
            "current_hypothesis": hypothesis,
            "feedback": hypothesis.pending_feedback,
        }

    def _is_full_pipeline(self, state: ResearchState) -> bool:
        """
        Cold-start (a brand-new research) runs the whole chain end to end.
        
        Args:
            state (ResearchState): Graph state.
        Returns:
            bool: Whether we are running a new research (full pipeline) or not.
        """
        return state["run_type"] == RunTypeEnum.NEW_RESEARCH

    def _intent_classifier(self, state: ResearchState) -> Dict[str, Any]:
        """
        Classify the latest user message into an intent. Skipped on cold-start
        (no real message to classify); falls back to CHAT on any failure so the
        graph degrades to a safe no-op instead of crashing.

        Args:
            state (ResearchState): Graph state.
        Returns:
            dict[str, Any]: {"intent": IntentEnum | None}.
        """
        if self._is_full_pipeline(state):
            log.info("cold start — skipping intent classification (full pipeline)")
            return {"intent": None}

        try:
            result = self.llm.invoke(
                prompt=CLASSIFY_INTENT_PROMPT,
                input={"message": state["messages"][-1].content},
                output_schema=IntentClassificationOutput,
            )
            data = result if isinstance(result, IntentClassificationOutput) else IntentClassificationOutput(**result.model_dump())
            log.info("classified intent: %s", data.intent.value)
            return {"intent": data.intent}

        except Exception:
            log.exception("intent classification failed — falling back to CHAT")
            return {"intent": IntentEnum.CHAT}

    def _intent_router(self, state: ResearchState) -> Command:
        """
        Route the turn: cold-start enters the hypothesizer (full pipeline); otherwise
        the intent picks an entry node, fanning out per-hypothesis for the
        literature/critique intents.

        Args:
            state (ResearchState): Graph state.
        Returns:
            Command: Redirection to the entry node(s).
        """
        if self._is_full_pipeline(state):
            log.info("routing → hypothesizer (full pipeline)")
            return Command(goto=NODE_HYPOTHESIZER, update={"active_agent": NODE_HYPOTHESIZER})

        target = INTENT_MAP.get(state["intent"], NODE_RESPONDER)

        if target in FANOUT_NODES:
            hypotheses = state["hypotheses"]
            if not hypotheses:
                log.info("intent %s needs hypotheses but none exist → responder", state["intent"])
                return Command(goto=NODE_RESPONDER, update={"active_agent": NODE_RESPONDER})
            log.info("routing → %s, fanning out over %d hypotheses", target, len(hypotheses))
            sends = [Send(target, self._branch_payload(state, h, pipeline=False)) for h in hypotheses]
            return Command(goto=sends, update={"active_agent": target})

        log.info("routing → %s", target)
        return Command(goto=target, update={"active_agent": target})

    def _hypothesizer_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Runs the hypothesizer sub-agent that handles hypotheses generation.

        Args:
            state (ResearchState): Graph state.
        Returns:
            dict[str, Any]: {"hypotheses": [...]}.
        """
        feedback = state.get("hypotheses_feedback")
        try:
            with log_stage(log, "hypothesis generation"):
                scored = self.hypothesizer.run(research_goal=state["goal"], feedback=feedback or "")
            hypotheses = [scored_to_hypothesis(scored=h) for h in scored]
            log.info("generated %d hypotheses%s", len(hypotheses), " (with feedback)" if feedback else "")

            return {"hypotheses": hypotheses, "hypotheses_feedback": None}
        except Exception:
            log.exception("hypothesizer node failed — no hypotheses produced")
            return {"hypotheses": [], "hypotheses_feedback": None}

    def _gate_decision(self, state: ResearchState, gate: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Pause for human approval at a gate, if that gate is armed for the active
        hitl_mode. Returns the human's decision dict, or None when the gate is not
        armed (the caller then proceeds without pausing).

        Args:
            state (ResearchState): Graph state.
            gate (str): One of GATE_HYPOTHESES / GATE_REFINE / GATE_REPORT.
            payload (Dict[str, Any]): JSON-serializable context shown to the human.
        Returns:
            Dict[str, Any] | None: {"action": "approve|reject|edit|feedback", "value": ...} or None.
        """
        if gate not in ARMED_GATES.get(state["hitl_mode"], set()):
            return None
        log.info("HITL gate '%s' armed (%s) — pausing for human", gate, state["hitl_mode"].value)
        return interrupt({"gate": gate, **payload})

    def _gate_hypotheses_node(self, state: ResearchState) -> Command:
        """
        HITL gate after hypothesis generation, before the literature fan-out.
        approve -> continue; reject -> stop (responder); edit -> apply per-id text/drop
        updates; feedback -> regenerate additional guided hypotheses.

        Args:
            state (ResearchState): Graph state.
        Returns:
            Command: Next hop, with any hypothesis/feedback updates.
        """
        decision = self._gate_decision(state, GATE_HYPOTHESES, {
            "question": "Review the generated hypotheses before literature review.",
            "hypotheses": [
                {"id": h.id, "text": h.text, "confidence": h.confidence, "status": h.status}
                for h in state["hypotheses"]
            ],
        })

        if decision is None or decision.get("action") == "approve":
            return Command(goto=NODE_HYPOTHESIZER_ROUTER)

        action = decision.get("action")
        if action == "reject":
            log.info("HITL: hypotheses rejected → responder")
            return Command(goto=NODE_RESPONDER, update={"active_agent": NODE_HYPOTHESIZER})

        if action == "edit":
            # value: [{"id": ..., "text"?: ..., "drop"?: bool}]; updates are same-id so
            # the merge reducer replaces them. Dropping parks the hypothesis.
            by_id = {h.id: h for h in state["hypotheses"]}
            updated = []
            for e in decision.get("value", []):
                h = by_id.get(e.get("id"))
                if h is None:
                    continue
                if e.get("drop"):
                    h.status = "parked"
                elif e.get("text"):
                    h.text = e["text"]
                updated.append(h)
            log.info("HITL: edited %d hypotheses", len(updated))
            return Command(goto=NODE_HYPOTHESIZER_ROUTER, update={"hypotheses": updated})

        if action == "feedback":
            log.info("HITL: feedback → regenerating additional hypotheses")
            return Command(goto=NODE_HYPOTHESIZER, update={"hypotheses_feedback": decision.get("value", "")})

        return Command(goto=NODE_HYPOTHESIZER_ROUTER)

    def _gate_report_node(self, state: ResearchState) -> Command:
        """
        HITL gate after the report is written, before the responder renders it.
        approve -> render as-is; edit -> render the human's verbatim text; feedback ->
        rewrite the report with guidance.

        Args:
            state (ResearchState): Graph state.
        Returns:
            Command: Next hop, with any report override / feedback updates.
        """
        report = state.get("report")
        decision = self._gate_decision(state, GATE_REPORT, {
            "question": "Review the research report before it is finalized.",
            "title": report.title if report else None,
            "report": self._render_report(report) if report else "(no report)",
        })

        if decision is None or decision.get("action") in (None, "approve"):
            return Command(goto=NODE_RESPONDER)

        action = decision.get("action")
        if action == "edit":
            log.info("HITL: report replaced with human edit")
            return Command(goto=NODE_RESPONDER, update={"report_override": decision.get("value", "")})

        if action == "feedback":
            log.info("HITL: feedback → rewriting report")
            return Command(goto=NODE_WRITER, update={"report_feedback": decision.get("value", "")})

        return Command(goto=NODE_RESPONDER)

    def _hypothesis_refiner_node(self, payload: Dict[str, Any]) -> Command[Literal["literature_reviewer", "responder"]]:
        """
        Per-hypothesis refine branch: take a parent hypothesis + its critique feedback,
        produce one re-scored child (lineage via parent_id), park the parent, and send
        the child back through the literature -> critic loop so it gets its own findings
        and a fresh verdict.

        Args:
            payload (Dict[str, Any]): Send branch payload (goal, current_hypothesis=parent, feedback).
        Returns:
            Command: Parent(parked)+child update, then fan the child into the literature reviewer.
        """
        try:
            parent = payload["current_hypothesis"]
            goal = payload["goal"]

            log.info("refining %s", _short(parent))
            with log_stage(log, f"refine {parent.id[:8]}"):
                scored_child = self.hypothesizer.refine(
                    research_goal=goal,
                    hypothesis=parent.text,
                    feedback=payload.get("feedback") or "",
                )
            child = scored_to_hypothesis(scored=scored_child, parent_id=parent.id)
            log.info("parked %s → child %s", parent.id[:8], _short(child))

            update = {"hypotheses": [park_parent(parent), child]}
            child_payload = {"goal": goal, "current_hypothesis": child, "pipeline": True}
            return Command(update=update, goto=Send(NODE_LITERATURE, child_payload))

        except Exception:
            log.exception("refine branch failed — skipping to responder")
            return Command(goto=NODE_RESPONDER)

    def _hypothesizer_router(self, state: ResearchState) -> Command[Literal["literature_reviewer", "responder"]]:
        """
        After generating: in the full pipeline, fan out the literature review over
        the hypotheses; otherwise stop and answer.

        Args:
            state (ResearchState): Graph state.
        Returns:
            Command: Fan-out to the literature reviewer, or hand off to the responder.
        """
        if not self._is_full_pipeline(state):
            return Command(goto=NODE_RESPONDER)

        hypotheses = [h for h in state["hypotheses"] if h.status == "active"]
        if not hypotheses:
            log.info("no active hypotheses to review → responder")
            return Command(goto=NODE_RESPONDER)

        log.info("fanning out literature review over %d hypotheses", len(hypotheses))
        sends = [Send(NODE_LITERATURE, self._branch_payload(state, h, pipeline=True)) for h in hypotheses]
        return Command(goto=sends, update={"active_agent": NODE_LITERATURE})

    def _literature_reviewer_node(self, payload: Dict[str, Any]) -> Command[Literal["critic", "responder"]]:
        """
        Per-hypothesis literature review branch: gather evidence for the hypothesis,
        stamp the returned findings with its id, and accumulate them. Continues to
        the critic in the pipeline, else hands off to the responder.

        Args:
            payload (Dict[str, Any]): Send branch payload (goal, current_hypothesis, pipeline).
        Returns:
            Command: Findings update + next hop (critic or responder).
        """
        try:
            hypothesis = payload["current_hypothesis"]
            goal = payload["goal"]

            log.info("literature review for %s", _short(hypothesis))
            with log_stage(log, f"literature review {hypothesis.id[:8]}"):
                results = self.literature_reviewer.run(
                    research_goal=goal,
                    hypothesis=hypothesis.text,
                )

            findings = stamp_findings(findings=results, hypothesis_id=hypothesis.id)
            log.info("%s → %d findings", hypothesis.id[:8], len(findings))
            update = {"findings": findings}

            if payload.get("pipeline"):
                return Command(
                    update=update,
                    goto=Send(NODE_CRITIC, {
                        "goal": payload["goal"],
                        "current_hypothesis": hypothesis,
                        "pipeline": True,
                    }),
                )
            return Command(update=update, goto=NODE_RESPONDER)

        except Exception:
            log.exception("literature review branch failed — skipping to responder")
            return Command(goto=NODE_RESPONDER)

    def _critic_node(self, payload: Dict[str, Any]) -> Command[Literal["critic_join", "responder"]]:
        """
        Per-hypothesis critique branch: assess the hypothesis adversarially and fold
        the verdict (+ feedback) into it via the adapter. In the pipeline it hands off
        to the critic-join (which decides loop-back vs writer); otherwise to the responder.

        Args:
            payload (Dict[str, Any]): Send branch payload (goal, current_hypothesis, pipeline).
        Returns:
            Command: Hypothesis update (merged by id) + next hop (critic_join or responder).
        """
        try:
            hypothesis = payload["current_hypothesis"]
            goal = payload["goal"]

            log.info("critique for %s", _short(hypothesis))
            with log_stage(log, f"critique {hypothesis.id[:8]}"):
                assessment = self.critic.run(
                    research_goal=goal,
                    hypothesis=hypothesis.text,
                )

            log.info("%s verdict: %s (falsifiability %.2f)",
                     hypothesis.id[:8], assessment.verdict, assessment.falsifiability_score)
            update = {"hypotheses": [apply_critic(hypothesis, assessment)]}
            goto = NODE_CRITIC_JOIN if payload.get("pipeline") else NODE_RESPONDER

            return Command(update=update, goto=goto)

        except Exception:
            log.exception("critique branch failed — skipping to responder")
            return Command(goto=NODE_RESPONDER)

    def _critic_join(self, state: ResearchState) -> Command[Literal["hypothesis_refiner", "literature_reviewer", "writer"]]:
        """
        Reduce step after the critic fan-out: bucket hypotheses by verdict and either
        loop back (refine / gather more evidence) or proceed to the writer. Bounded by
        REFINE_CAP so the loop always terminates.

          needs_refinement     -> refiner (produces a child, re-enters literature->critic)
          needs_more_evidence  -> literature reviewer (same hypothesis, re-gather)
          holds / refuted      -> left in state for the writer
          round >= REFINE_CAP  -> force writer regardless of remaining verdicts

        Args:
            state (ResearchState): Graph state (merged across critic branches).
        Returns:
            Command: Loop-back Sends (+ round bump) or hand off to the writer.
        """
        round = state.get("refine_round", 0)

        verdicts = [h.pending_verdict for h in state["hypotheses"]]
        counts = {v: verdicts.count(v) for v in set(verdicts)}
        log.info("critic join (round %d/%d) — verdicts: %s", round, REFINE_CAP, counts)

        sends = []
        if round < REFINE_CAP:
            for h in state["hypotheses"]:
                if h.pending_verdict == "needs_refinement":
                    sends.append(Send(NODE_HYPOTHESIS_REFINER, self._refine_payload(state, h)))
                elif h.pending_verdict == "needs_more_evidence":
                    sends.append(Send(NODE_LITERATURE, self._branch_payload(state, h, pipeline=True)))

        if sends:
            decision = self._gate_decision(state, GATE_REFINE, {
                "question": f"Critic wants another refine round ({len(sends)} hypotheses). Continue?",
                "round": round,
                "pending": counts,
            })
            if decision is not None and decision.get("action") == "reject":
                log.info("HITL: refine loop stopped by human → writer")
                return Command(goto=NODE_WRITER)

            log.info("looping back %d hypotheses (round → %d)", len(sends), round + 1)
            return Command(goto=sends, update={"refine_round": round + 1})

        reason = "cap reached" if round >= REFINE_CAP else "all verdicts settled"
        log.info("no loop-back (%s) → writer", reason)
        return Command(goto=NODE_WRITER)

    async def _writer_node(self, state: ResearchState) -> Command[Literal["responder"]]:
        """
        Join after the fan-out: resolve citation sources for the cited papers, run the
        Writer over the accumulated hypotheses + findings, and stash the report in
        state for the responder to render.

        Args:
            state (ResearchState): Graph state (merged across branches).
        Returns:
            Command: Report update + hand off to the responder.
        """
        feedback = state.get("report_feedback")
        try:
            paper_ids = list({f.source_paper_id for f in state["findings"] if f.source_paper_id})
            rows = await load_paper_sources(paper_ids)
            sources = [SourceMeta(**row) for row in rows]
            log.info("writing report: %d hypotheses, %d findings, %d sources%s",
                     len(state["hypotheses"]), len(state["findings"]), len(sources),
                     " (with feedback)" if feedback else "")

            with log_stage(log, "report writing"):
                report = self.writer.run(
                    research_goal=state["goal"],
                    hypotheses=state["hypotheses"],
                    findings=state["findings"],
                    sources=sources,
                    feedback=feedback or "",
                )

            log.info("report produced: %s", report.title if report else "(none)")
            return Command(goto=NODE_GATE_REPORT, update={"report": report, "report_feedback": None, "active_agent": NODE_WRITER})
        except Exception:
            log.exception("writer node failed — skipping to responder")
            return Command(goto=NODE_RESPONDER)

    @staticmethod
    def _render_report(report: ReportOutput) -> str:
        """
        Render a structured report into the markdown shown to the human.

        Args:
            report (ReportOutput): The Writer's structured report.
        Returns:
            str: Markdown rendering of the report.
        """
        lines = [f"# {report.title}", "", report.introduction, "", "## Key findings", report.key_findings]

        if report.hypotheses_assessment:
            lines += ["", "## Hypotheses"]
            for v in report.hypotheses_assessment:
                lines.append(f"- **[{v.status}]** {v.hypothesis} — {v.assessment}")

        if report.open_questions:
            lines += ["", "## Open questions"] + [f"- {q}" for q in report.open_questions]

        lines += ["", "## Conclusion", report.conclusion]

        if report.references:
            lines += ["", "## References"]
            for c in report.references:
                suffix = f" ({c.url})" if c.url else ""
                lines.append(f"[{c.number}] {c.title or c.paper_id}{suffix}")

        return "\n".join(lines)

    def _responder_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Single convergence point: turns the turn's outcome into one AI message. When a
        report was produced this turn it is rendered in full; otherwise an LLM pass
        answers from the goal, hypotheses, findings, recent messages, and summary.

        Args:
            state (ResearchState): Graph state.
        Returns:
            dict[str, Any]: One appended AI message; clears active_agent.
        """
        try:
            report = state.get("report")
            override = state.get("report_override")
            if state.get("active_agent") == NODE_WRITER and override:
                log.info("responder: rendering human-edited report")
                text = override
            elif state.get("active_agent") == NODE_WRITER and report is not None:
                log.info("responder: rendering report")
                text = self._render_report(report)
            else:
                log.info("responder: generating conversational reply (last agent: %s)", state.get("active_agent"))
                result = self.llm.invoke(
                    prompt=RESPOND_PROMPT,
                    input={
                        "goal": state["goal"],
                        "hypotheses": [h.model_dump() for h in state["hypotheses"]],
                        "findings": [f.model_dump() for f in state["findings"]],
                        "messages": state["messages"][-CONTEXT_MESSAGES:],
                        "summary": state["session_summary"]
                        },
                    output_schema=RespondOutput,
                )
                data = result if isinstance(result, RespondOutput) else RespondOutput(**result.model_dump())
                text = data.text
            
            return {"messages": [AIMessage(content=text)], "active_agent": None}

        except Exception:
            log.exception("responder failed — emitting fallback message")
            return {"messages": [AIMessage(content="Something went wrong when answering.")], "active_agent": None}

    def _summary_router(self, state: ResearchState) -> Command[Literal["summarization", "persist"]]:
        """
        Summarize the rolling session memory once enough new messages have
        accumulated since the last summary; otherwise go straight to persist.

        Args:
            state (ResearchState): Graph state.
        Returns:
            Command: To the summarizer or directly to persist.
        """
        grown = len(state["messages"]) - (state.get("last_summarized_at") or 0)
        if grown >= SUMMARY_EVERY:
            log.info("summary: %d new messages ≥ %d → summarizing", grown, SUMMARY_EVERY)
            return Command(goto=NODE_SUMMARIZER)
        return Command(goto=NODE_PERSIST)

    def _summarization_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Refresh the rolling session summary by folding the recent message window into
        the prior summary, and advance the last_summarized_at watermark.

        Args:
            state (ResearchState): Graph state.
        Returns:
            dict[str, Any]: Updated summary + last_summarized_at watermark.
        """
        with log_stage(log, "summarization"):
            summary = generate_summary_memory(crr_summary=state["session_summary"] or "",
                                    messages=state["messages"])

        return {
            "session_summary": summary,
            "last_summarized_at": len(state["messages"]),
        }

    async def _persist_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Terminal persistence: write the turn (summary, hypotheses, findings,
        new messages) back to Postgres.

        Args:
            state (ResearchState): Graph state.
        Returns:
            dict[str, Any]: Empty (no state change).
        """
        log.info("persisting turn: %d hypotheses, %d findings, %d messages",
                 len(state["hypotheses"]), len(state["findings"]), len(state["messages"]))
        await save_state(state)
        log.info("persisted")
        return {}

    async def run(self,
                  mode: HumanInTheLoopEnum,
                  goal: str | None = None,
                  session_id: str | None = None,
                  research_id: str | None = None,
                  message: str | None = None):
        """
        Bootstrap the state (new research / resumed conversation / new session on an
        existing research), inject the per-turn message, and run one supervisor turn.

        Args:
            mode (HumanInTheLoopEnum): Human-in-the-loop mode.
            goal (str | None): New research goal (cold start).
            session_id (str | None): Resume this conversation.
            research_id (str | None): Open a new session on this existing research.
            message (str | None): The user's instruction for this turn.
        Returns:
            dict: A turn result — see `_finish`. Paused turns carry the interrupt
                payload + thread_id so the caller can collect input and `resume`.
        """
        if goal:
            run_type = RunTypeEnum.NEW_RESEARCH
            state = await create_research(goal, mode)
            log.info("NEW RESEARCH (mode=%s) goal=%r [research=%s session=%s]",
                     mode.value, goal, state["research_id"], state["session_id"])
        elif session_id:
            run_type = RunTypeEnum.RESUME_CONVERSATION
            state = await load_state(session_id)
            log.info("RESUME CONVERSATION (mode=%s) [session=%s]", mode.value, session_id)
        elif research_id:
            run_type = RunTypeEnum.NEW_SESSION_IN_RESEARCH
            state = await start_session_in_research(research_id, mode)
            log.info("NEW SESSION IN RESEARCH (mode=%s) [research=%s session=%s]",
                     mode.value, research_id, state["session_id"])
        else:
            raise ValueError("Missing 'goal' or 'session_id' or 'research_id'.")

        self.searcher.reset_ingestion_budget()

        state["run_type"] = run_type
        state["intent"] = None
        state["active_agent"] = None
        if message:
            log.info("user message: %r", message)
            state["messages"].append(HumanMessage(content=message))

        thread_id = str(uuid4())
        sid = state["session_id"]
        with log_stage(log, "turn"):
            result = await self.graph.ainvoke(state, self._config(thread_id))
        return await self._finish(result, sid, thread_id)

    async def resume(self, session_id: str, thread_id: str, decision: Dict[str, Any]):
        """
        Resume a turn paused at a HITL gate, feeding the human's decision back in.

        Args:
            session_id (str): The session that owns the paused turn.
            thread_id (str): The paused turn's thread_id (from the run/resume result).
            decision (Dict[str, Any]): {"action": "approve|reject|edit|feedback", "value"?: ...}.
        Returns:
            dict: A turn result — may be paused again (another gate) or done.
        """
        log.info("resuming thread %s with decision %s", thread_id, decision.get("action"))
        result = await self.graph.ainvoke(Command(resume=decision), self._config(thread_id))
        return await self._finish(result, session_id, thread_id)

    @staticmethod
    def _config(thread_id: str) -> Dict[str, Any]:
        """
        LangGraph invocation config keyed by thread_id (= one turn's pause/resume).
        """
        return {"configurable": {"thread_id": thread_id}}

    async def _finish(self, result: Dict[str, Any], session_id: str, thread_id: str) -> Dict[str, Any]:
        """
        Normalize an ainvoke result into a turn result, and track the pending pause on
        the session so it can be resumed after a restart / from a UI.

        Args:
            result (Dict[str, Any]): The graph's ainvoke return.
            session_id (str): Session that owns the turn.
            thread_id (str): The turn's thread_id.
        Returns:
            dict: {"paused": bool, "thread_id", "session_id", "state", "interrupt"?}.
        """
        if "__interrupt__" in result:
            await set_pending_thread(session_id, thread_id)
            payload = result["__interrupt__"][0].value
            log.info("turn paused at gate '%s'", payload.get("gate"))
            return {"paused": True, "thread_id": thread_id, "session_id": session_id,
                    "interrupt": payload, "state": result}

        await set_pending_thread(session_id, None)
        return {"paused": False, "thread_id": thread_id, "session_id": session_id, "state": result}
