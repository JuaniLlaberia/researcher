from typing import List, Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """
    A single factual claim extracted from the literature, with its stance toward the hypothesis.
    """
    content: str = Field(..., description="A single, self-contained factual claim drawn from the evidence, stated in one or two sentences. Must be grounded in the retrieved text.")
    stance: Literal["supports", "contradicts", "neutral"] = Field(..., description="The claim's relationship to the hypothesis: 'supports' if it is evidence for it, 'contradicts' if it is evidence against it, 'neutral' if it is relevant context but neither.")
    source_paper_id: str | None = Field(None, description="The paper_id of the evidence item this claim came from, for source attribution. Null only if it cannot be tied to a specific paper.",)

class QueriesOutput(BaseModel):
    main_query: str = Field(..., description="The single primary search query: a concise, information-dense reformulation of what must be found to test the hypothesis. Used for keyword search and as the rerank anchor.")
    queries: List[str] = Field(..., min_length=1, max_length=3, description="1 to 3 semantically diverse variants of the main query (synonyms, narrower/broader phrasings, different angles) to widen retrieval recall.")

class DataValidatorOutput(BaseModel):
    validation_label: Literal["sufficient", "insufficient", "needs_different_queries"] = Field(
        ...,
        description="'sufficient': evidence is enough to extract meaningful findings."
                    "'insufficient': on-topic but too sparse/shallow, more of the same is needed."
                    "'needs_different_queries': evidence does not address the hypothesis, a different search angle is needed.",
    )
    feedback: str = Field(..., description="Actionable guidance for the query generator on what is missing or what angle to try next. Empty string when label is 'sufficient'.")

class FindingExtractorOutput(BaseModel):
    findings: List[Finding] = Field(..., description="All distinct findings extracted from the evidence. May be empty if the evidence contains no claims relevant to the hypothesis.")

class EvidenceItem(BaseModel):
    source_type: str = Field(..., description="Where the evidence came from: 'vector_db' (full-text chunk), 'arxiv', or 'semantic_scholar'.")
    title: str | None = Field(None, description="Title of the source paper, if known.")
    url: str | None = Field(None, description="Link to the source paper, if known.")
    text: str = Field(..., description="The chunk text (or abstract fallback) that constitutes the evidence.")
    paper_id: str | None = Field(None, description="The stored paper_id this chunk belongs to, used for source attribution.")
    score: float | None = Field(None, description="Retrieval relevance (reranker) score, if available.")

class RelevanceGateOutput(BaseModel):
    """
    Which discovered candidates are worth ingesting (by list index).
    """
    selected_indices: List[int] = Field(default_factory=list, description="The `index` values of the candidate papers worth ingesting, chosen by relevance to the hypothesis. Empty if none justify ingestion.")
