import enum
from pydantic import BaseModel, Field

class IntentEnum(str, enum.Enum):
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    REVIEW_LITERATURE = "review_literature"
    CRITIQUE_HYPOTHESIS = "critique_hypothesis"
    RESEARCH_REPORT = "research_report"
    CHAT = "chat"

class IntentClassificationOutput(BaseModel):
    intent: IntentEnum = Field(..., description="Detected intent in user message.")

class RespondOutput(BaseModel):
    text: str = Field(..., description="Message answering user.")