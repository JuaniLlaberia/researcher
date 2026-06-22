import enum
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

from src.researcher.models import Hypothesis, Finding
from src.researcher.utils.models import IntentEnum
from src.researcher.reducers import merge_hypotheses, merge_findings
from src.researcher.agents.writer.utils.models import ReportOutput

class HumanInTheLoopEnum(enum.Enum):
    INTERACTIVE = "interactive" # agent asks for human approval before every significant action
    CHECKPOINT = "checkpoint" # agent runs until a natural decision point, then pauses and waits for human input
    AUTONOMOUS = "autonomous" # agent runs freely until it hits a budget limit or hard error

class RunTypeEnum(enum.Enum):
    NEW_RESEARCH = "new_research"
    RESUME_CONVERSATION = "resume_conversation"
    NEW_SESSION_IN_RESEARCH = "new_session_in_research"

class ResearchState(TypedDict):
    research_id: str
    goal: str
    hypotheses: Annotated[List[Hypothesis], merge_hypotheses]

    findings: Annotated[List[Finding], merge_findings]
    ingested_paper_ids: List[str]
    decision_trail: List[dict]

    report: ReportOutput | None  # latest report from the Writer; rendered by the responder, not persisted as a structured row

    session_id: str
    session_summary: str | None
    last_summarized_at: int

    hitl_mode: HumanInTheLoopEnum
    run_type: RunTypeEnum
    intent: IntentEnum | None
    refine_round: int            # transient: refinement passes taken this run; bounds the critic refine loop

    messages: Annotated[list, add_messages]
    active_agent: str | None
