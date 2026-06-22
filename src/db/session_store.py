from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import func, select

from src.db.models.finding import Finding as FindingModel
from src.db.models.finding import Stance
from src.db.models.hypothesis import Hypothesis as HypothesisModel
from src.db.models.hypothesis import HypothesisStatus
from src.db.models.message import Message, Role
from src.db.models.paper import Paper
from src.db.models.research import Research, ResearchStatus
from src.db.models.session import Session
from src.db.session import AsyncSessionLocal
from src.core.logging import get_logger
from src.researcher.models import Finding
from src.researcher.state import Hypothesis, HumanInTheLoopEnum, ResearchState

DEFAULT_WINDOW = 15

log = get_logger("db")

def _empty_state(
    research_id: str,
    goal: str,
    session_id: str,
    hitl_mode: HumanInTheLoopEnum,
    *,
    session_summary: str | None = None,
    hypotheses: list[Hypothesis] | None = None,
    findings: list[Finding] | None = None,
    ingested_paper_ids: list[str] | None = None,
    messages: list[BaseMessage] | None = None,
) -> ResearchState:
    """
    Build a fully-keyed ResearchState. `run_type` / `intent` are runtime-only and
    left as None here; the orchestrator's `run()` sets them before invoking.
    """
    return ResearchState(
        research_id=research_id,
        goal=goal,
        hypotheses=hypotheses or [],
        findings=findings or [],
        ingested_paper_ids=ingested_paper_ids or [],
        report=None,
        session_id=session_id,
        session_summary=session_summary,
        last_summarized_at=0,
        hitl_mode=hitl_mode,
        run_type=None,
        intent=None,
        refine_round=0,
        hypotheses_feedback=None,
        report_feedback=None,
        report_override=None,
        messages=messages or [],
        active_agent=None,
    )

async def load_paper_sources(paper_ids: list[str]) -> list[dict]:
    """
    Resolve source metadata (title/url) for a set of paper ids, for the Writer's
    citations. A finding's `source_paper_id` is the `papers.id` UUID (set by the
    vector store), so this is a direct lookup. Unparseable ids are skipped.

    Args:
        paper_ids (list[str]): Distinct paper ids referenced by the findings.
    Returns:
        list[dict]: [{"paper_id": str, "title": str | None, "url": str | None}].
    """
    uuids = []
    for pid in paper_ids:
        try:
            uuids.append(UUID(pid))
        except (ValueError, TypeError):
            continue
    if not uuids:
        return []

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(Paper.id, Paper.title, Paper.url).where(Paper.id.in_(uuids))
            )
        ).all()

    return [{"paper_id": str(pid), "title": title, "url": url} for pid, title, url in rows]

def _row_to_hypothesis(row: HypothesisModel) -> Hypothesis:
    """
    hypotheses row -> Hypothesis TypedDict (note: `score` column <-> `confidence`).
    """
    return Hypothesis(
        id=str(row.id),
        text=row.text,
        confidence=row.score if row.score is not None else 0.0,
        falsifiability_score=row.falsifiability_score,
        parent_id=str(row.parent_id) if row.parent_id else None,
        status=row.status.value,
    )

def _row_to_finding(row: FindingModel) -> Finding:
    """
    findings row -> Finding domain model (paper_id column <-> source_paper_id).
    """
    return Finding(
        id=str(row.id),
        content=row.content,
        stance=row.stance.value,
        source_paper_id=str(row.paper_id) if row.paper_id else None,
        hypothesis_id=str(row.hypothesis_id) if row.hypothesis_id else None,
    )

async def _load_research_artifacts(s, research_id: UUID) -> tuple[list[Hypothesis], list[Finding], list[str]]:
    """
    Load the project-level artifacts shared across sessions: hypotheses, findings,
    and ingested paper ids for a research.
    """
    hyp_rows = (
        await s.execute(
            select(HypothesisModel).where(HypothesisModel.research_id == research_id)
        )
    ).scalars().all()

    finding_rows = (
        await s.execute(
            select(FindingModel).where(FindingModel.research_id == research_id)
        )
    ).scalars().all()

    paper_ids = (
        await s.execute(select(Paper.id).where(Paper.research_id == research_id))
    ).scalars().all()

    return (
        [_row_to_hypothesis(r) for r in hyp_rows],
        [_row_to_finding(r) for r in finding_rows],
        [str(pid) for pid in paper_ids],
    )

async def create_research(goal: str, hitl_mode: HumanInTheLoopEnum) -> ResearchState:
    """
    Bootstrap a brand-new research: create the `researchs` row and its first
    `sessions` row, and return a fresh, fully-keyed state.

    Args:
        goal (str): The research goal.
        hitl_mode (HumanInTheLoopEnum): Human-in-the-loop mode for the session.
    Returns:
        ResearchState: Empty state wired to the new research + session ids.
    """
    async with AsyncSessionLocal() as s:
        async with s.begin():
            research = Research(goal=goal, status=ResearchStatus.ACTIVE)
            s.add(research)
            await s.flush()
            session = Session(research_id=research.id, hitl_mode=hitl_mode.value)
            s.add(session)
            await s.flush()
            rid, sid = str(research.id), str(session.id)

    log.info("created research %s + session %s", rid, sid)
    return _empty_state(rid, goal, sid, hitl_mode)

async def start_session_in_research(research_id: str, hitl_mode: HumanInTheLoopEnum) -> ResearchState:
    """
    Open a new session under an existing research: a fresh conversation (empty
    message window + no summary) over the project's accumulated hypotheses and
    findings.

    Args:
        research_id (str): The research to continue.
        hitl_mode (HumanInTheLoopEnum): Human-in-the-loop mode for the new session.
    Returns:
        ResearchState: State hydrated with project artifacts, empty conversation.
    """
    rid = UUID(research_id)

    async with AsyncSessionLocal() as s:
        async with s.begin():
            research = await s.get(Research, rid)
            if research is None:
                raise ValueError(f"No research with id {research_id}")

            session = Session(research_id=rid, hitl_mode=hitl_mode.value)
            s.add(session)
            await s.flush()
            sid = str(session.id)

            hypotheses, findings, paper_ids = await _load_research_artifacts(s, rid)
            goal = research.goal

    return _empty_state(
        research_id,
        goal,
        sid,
        hitl_mode,
        hypotheses=hypotheses,
        findings=findings,
        ingested_paper_ids=paper_ids,
    )

async def set_pending_thread(session_id: str, thread_id: str | None) -> None:
    """
    Record (or clear) the LangGraph thread_id of a turn paused at a HITL gate, so a
    paused run can be located and resumed after a restart or from a UI.

    Args:
        session_id (str): The session whose pending pause is being tracked.
        thread_id (str | None): The paused turn's thread_id, or None to clear it.
    """
    async with AsyncSessionLocal() as s:
        async with s.begin():
            session = await s.get(Session, UUID(session_id))
            if session is None:
                raise ValueError(f"No session with id {session_id}")
            session.pending_thread_id = thread_id

def _row_to_message(row: Message) -> BaseMessage:
    """
    messages row -> LangChain message (USER -> Human, AGENT -> AI).
    """
    if row.role is Role.USER:
        return HumanMessage(content=row.content)
    return AIMessage(content=row.content)

def _message_role(msg: BaseMessage) -> Role:
    """
    LangChain message -> messages.role enum (Human -> USER, everything else -> AGENT).
    """
    return Role.USER if isinstance(msg, HumanMessage) else Role.AGENT

async def load_state(session_id: str, window_size: int = DEFAULT_WINDOW) -> ResearchState:
    """
    Hydrate a `ResearchState` from the tables for the given session.

    Loads the session + its research, all hypotheses for that research, and the
    most recent `window_size` messages (in chronological order).

    Args:
        session_id (str): Session ID to be retrieve.
        window_size (int): Amount of messages to put in context.
    Returns:
        ResearchState: State ready to be use in the langgraph graph.
    """
    sid = UUID(session_id)

    async with AsyncSessionLocal() as s:
        session = await s.get(Session, sid)
        if session is None:
            raise ValueError(f"No session with id {session_id}")

        research = await s.get(Research, session.research_id)

        hypotheses, findings, paper_ids = await _load_research_artifacts(s, session.research_id)

        # Most recent N by seq, then flip back to chronological order.
        msg_rows = (
            await s.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.seq.desc())
                .limit(window_size)
            )
        ).scalars().all()
        msg_rows = list(reversed(msg_rows))

    log.info("loaded session %s: %d hypotheses, %d findings, %d messages",
             session_id, len(hypotheses), len(findings), len(msg_rows))
    return _empty_state(
        str(session.research_id),
        research.goal,
        str(session.id),
        HumanInTheLoopEnum(session.hitl_mode),
        session_summary=session.summary,
        hypotheses=hypotheses,
        findings=findings,
        ingested_paper_ids=paper_ids,
        messages=[_row_to_message(r) for r in msg_rows],
    )

async def save_state(state: ResearchState, window_size: int = DEFAULT_WINDOW) -> None:
    """
    Persist the mutable parts of `state` back to the tables, in one transaction:

      * sessions: summary, hitl_mode.
      * hypotheses: upsert by id (update if present, insert if new).
      * messages: append only the messages produced during this run.

    Args:
        state (ResearchState): Graph state to save.
        window_size (int): Amount of messages to save.
    Returns:
        None.
    """
    sid = UUID(state["session_id"])
    rid = UUID(state["research_id"])

    async with AsyncSessionLocal() as s:
        async with s.begin():
            session = await s.get(Session, sid)
            if session is None:
                raise ValueError(f"No session with id {state['session_id']}")
            session.summary = state["session_summary"]
            session.hitl_mode = state["hitl_mode"].value

            for h in state["hypotheses"]:
                hid = UUID(h.id)
                row = await s.get(HypothesisModel, hid)
                if row is None:
                    s.add(
                        HypothesisModel(
                            id=hid,
                            text=h.text,
                            score=h.confidence,
                            falsifiability_score=h.falsifiability_score,
                            status=HypothesisStatus(h.status),
                            research_id=rid,
                            parent_id=UUID(h.parent_id) if h.parent_id else None,
                        )
                    )
                else:
                    row.text = h.text
                    row.score = h.confidence
                    row.falsifiability_score = h.falsifiability_score
                    row.status = HypothesisStatus(h.status)
                    row.parent_id = UUID(h.parent_id) if h.parent_id else None

            for f in state["findings"]:
                if not f.source_paper_id or not f.hypothesis_id:
                    continue
                try:
                    fid, pid, hid = UUID(f.id), UUID(f.source_paper_id), UUID(f.hypothesis_id)
                except ValueError:
                    continue
                if await s.get(FindingModel, fid) is None:
                    s.add(
                        FindingModel(
                            id=fid,
                            content=f.content,
                            stance=Stance(f.stance),
                            research_id=rid,
                            paper_id=pid,
                            hypothesis_id=hid,
                        )
                    )

            db_count = await s.scalar(
                select(func.count()).select_from(Message).where(Message.session_id == sid)
            )
            already_loaded = min(window_size, db_count)
            for msg in state["messages"][already_loaded:]:
                s.add(
                    Message(
                        role=_message_role(msg),
                        content=msg.content,
                        session_id=sid,
                        research_id=rid,
                    )
                )
