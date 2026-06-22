from typing import List, Literal
from pydantic import BaseModel, Field

class AdversarialQueriesOutput(BaseModel):
    queries: List[str] = Field(..., description="Search queries aimed at finding evidence that contradicts, weakens, or exposes flaws in the hypothesis (not evidence that supports it).")

class AdversarialRelevanceGateOutput(BaseModel):
    selected_indices: List[int] = Field(default_factory=list, description="The `index` values of the candidate papers most likely to contradict the hypothesis or reveal its flaws, chosen for ingestion. Empty if none are worth ingesting.")

class CriticAssessment(BaseModel):
    contradictions: list[str] = Field(..., description="Specific points where the gathered evidence contradicts or undermines the hypothesis. Empty if no contradicting evidence was found.")
    assumptions: list[str] = Field(..., description="Hidden or unstated assumptions the hypothesis depends on, which may not hold.")
    falsifiability_score: float = Field(..., ge=0, le=1, description="How concretely testable/refutable the hypothesis is. 1 = a clear, falsifiable prediction; 0 = unfalsifiable.")
    rationale: str = Field(..., description="One or two sentences justifying the verdict, weighing supporting vs contradicting evidence.")
    verdict: Literal["holds", "refuted", "needs_refinement", "needs_more_evidence"] = Field(
        ...,
        description="'holds': survives adversarial scrutiny. "
                    "'refuted': contradicting evidence outweighs support. "
                    "'needs_refinement': promising but has a flaw or shaky assumption to address. "
                    "'needs_more_evidence': too little evidence to judge.",
    )