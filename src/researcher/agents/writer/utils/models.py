from typing import List
from pydantic import BaseModel, Field

class SourceMeta(BaseModel):
    paper_id: str = Field(..., description="The stored paper_id, matched against each finding's source_paper_id.")
    title: str | None = Field(None, description="Title of the source paper, if known.")
    url: str | None = Field(None, description="Link to the source paper, if known.")

class Citation(BaseModel):
    number: int = Field(..., description="The citation number used by inline [n] markers in the prose.")
    paper_id: str = Field(..., description="The paper_id this citation points to.")
    title: str | None = Field(None, description="Title of the source paper, if known.")
    url: str | None = Field(None, description="Link to the source paper, if known.")

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="A short executive abstract of the research so far: the goal, the state of the hypotheses, and the headline takeaways from the findings.")

class LiteratureTheme(BaseModel):
    theme: str = Field(..., description="A specific topic or sub-question the findings cluster around.")
    synthesis: str = Field(..., description="A narrative tying the related findings together, noting where they agree and where they disagree.")
    source_paper_ids: List[str] = Field(default_factory=list, description="paper_ids of the findings this theme draws on, for source attribution.")

class LiteratureAnalysisOutput(BaseModel):
    themes: List[LiteratureTheme] = Field(..., description="The body of evidence organized into coherent themes rather than paper-by-paper.")
    consensus: List[str] = Field(default_factory=list, description="Points that are well-supported across multiple findings.")
    contradictions: List[str] = Field(default_factory=list, description="Points where findings conflict with one another.")
    gaps: List[str] = Field(default_factory=list, description="Questions the research goal raises that the gathered evidence does not address.")

class HypothesisVerdict(BaseModel):
    hypothesis: str = Field(..., description="The hypothesis text.")
    status: str = Field(..., description="Its final status (e.g. active, supported, refuted, parked).")
    assessment: str = Field(..., description="One or two sentences on where the hypothesis stands and why, citing the evidence.")

class ReportOutput(BaseModel):
    title: str = Field(..., description="A concise title for the research report.")
    introduction: str = Field(..., description="The research goal and what was explored.")
    hypotheses_assessment: List[HypothesisVerdict] = Field(..., description="Per-hypothesis outcome.")
    key_findings: str = Field(..., description="Narrative of what was found, drawn from the literature analysis.")
    open_questions: List[str] = Field(default_factory=list, description="What remains unresolved or worth pursuing next.")
    conclusion: str = Field(..., description="A short closing synthesis.")
    references: List[Citation] = Field(default_factory=list, description="Numbered references for the inline [n] citations. Populated deterministically by the agent, not by the model.")
