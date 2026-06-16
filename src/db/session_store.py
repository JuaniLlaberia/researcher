from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import func, select

from src.db.models.hypothesis import Hypothesis as HypothesisModel
from src.db.models.hypothesis import HypothesisStatus
from src.db.models.message import Message, Role
from src.db.models.research import Research
from src.db.models.session import Session
from src.db.session import AsyncSessionLocal
from src.researcher.state import Hypothesis, HumanInTheLoopEnum, ResearchState

DEFAULT_WINDOW = 15

def _row_to_hypothesis(row: HypothesisModel) -> Hypothesis:
    """
    hypotheses row -> Hypothesis TypedDict (note: `score` column <-> `confidence`).
    """
    return Hypothesis(
        id=str(row.id),
        text=row.text,
        confidence=row.score,
        parent_id=str(row.parent_id) if row.parent_id else None,
        status=row.status.value,
    )

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

        hyp_rows = (
            await s.execute(
                select(HypothesisModel).where(
                    HypothesisModel.research_id == session.research_id
                )
            )
        ).scalars().all()

        # Most recent N by created_at, then flip back to chronological order.
        msg_rows = (
            await s.execute(
                select(Message)
                .where(Message.session_id == sid)
                .order_by(Message.seq.desc())
                .limit(window_size)
            )
        ).scalars().all()
        msg_rows = list(reversed(msg_rows))

    return ResearchState(
        research_id=str(session.research_id),
        goal=research.goal,
        hypotheses=[_row_to_hypothesis(r) for r in hyp_rows],
        session_id=str(session.id),
        session_summary=session.summary,
        hitl_mode=HumanInTheLoopEnum(session.hitl_mode),
        messages=[_row_to_message(r) for r in msg_rows],
        active_agent=None,
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
                hid = UUID(h["id"])
                row = await s.get(HypothesisModel, hid)
                if row is None:
                    s.add(
                        HypothesisModel(
                            id=hid,
                            text=h["text"],
                            score=h["confidence"],
                            status=HypothesisStatus(h["status"]),
                            research_id=rid,
                            parent_id=UUID(h["parent_id"]) if h["parent_id"] else None,
                        )
                    )
                else:
                    row.text = h["text"]
                    row.score = h["confidence"]
                    row.status = HypothesisStatus(h["status"])
                    row.parent_id = UUID(h["parent_id"]) if h["parent_id"] else None

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
