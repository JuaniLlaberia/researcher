from typing import List, Literal
from pydantic import BaseModel, Field

from src.researcher.models import Finding

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

class RelevanceGateOutput(BaseModel):
    """
    Which discovered candidates are worth ingesting (by list index).
    """
    selected_indices: List[int] = Field(default_factory=list, description="The `index` values of the candidate papers worth ingesting, chosen by relevance to the hypothesis. Empty if none justify ingestion.")
