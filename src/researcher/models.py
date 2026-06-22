from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, Field

HypothesisStatus = Literal[
    "pending", "active", "testing", "supported", "refuted", "parked"
]

class EvidenceItem(BaseModel):
    source_type: str = Field(..., description="Where the evidence came from: 'vector_db' (full-text chunk), 'arxiv', or 'semantic_scholar'.")
    chunk_id: str | None = Field(None, description="The retrieved chunk's id, used to deduplicate evidence accumulated across search loops.")
    title: str | None = Field(None, description="Title of the source paper, if known.")
    url: str | None = Field(None, description="Link to the source paper, if known.")
    text: str = Field(..., description="The chunk text (or abstract fallback) that constitutes the evidence.")
    paper_id: str | None = Field(None, description="The stored paper_id this chunk belongs to, used for source attribution.")
    score: float | None = Field(None, description="Retrieval relevance (reranker) score, if available.")

class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), description="Stable hypothesis id (uuid) shared with the hypotheses table.")
    text: str = Field(..., description="The hypothesis statement.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Composite confidence from the Hypothesizer scorer. Maps to hypotheses.score.")
    status: HypothesisStatus = Field("active", description="Lifecycle status. Set from the Critic verdict during a run.")
    parent_id: str | None = Field(None, description="The hypothesis this one was derived from, if any.")
    falsifiability_score: float | None = Field(None, ge=0.0, le=1.0, description="How concretely testable/refutable the hypothesis is, produced by the Critic.")
    pending_verdict: Literal["holds", "refuted", "needs_refinement", "needs_more_evidence"] | None = Field(
        None, description="Last Critic verdict for this hypothesis; transient routing signal for the refine loop, not persisted."
    )
    pending_feedback: str | None = Field(
        None, description="Condensed Critic critique (contradictions/assumptions/rationale) fed to the refine step; transient, not persisted."
    )

class Finding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), description="Stable finding id (uuid) shared with the findings table, used to persist/dedup by id.")
    content: str = Field(..., description="A single, self-contained factual claim drawn from the evidence, stated in one or two sentences. Must be grounded in the retrieved text.")
    stance: Literal["supports", "contradicts", "neutral"] = Field(..., description="The claim's relationship to the hypothesis: 'supports' if it is evidence for it, 'contradicts' if it is evidence against it, 'neutral' if it is relevant context but neither.")
    source_paper_id: str | None = Field(None, description="The paper_id of the evidence item this claim came from, for source attribution. Null only if it cannot be tied to a specific paper.")
    hypothesis_id: str | None = Field(None, description="The hypothesis this finding was gathered for. Stamped by the orchestrator after the Literature Reviewer returns.")
