from pydantic import BaseModel, Field

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="Summary of the current session based on the messages and prior summary.")