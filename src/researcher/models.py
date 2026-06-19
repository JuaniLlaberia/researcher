from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    source_type: str = Field(..., description="Where the evidence came from: 'vector_db' (full-text chunk), 'arxiv', or 'semantic_scholar'.")
    chunk_id: str | None = Field(None, description="The retrieved chunk's id, used to deduplicate evidence accumulated across search loops.")
    title: str | None = Field(None, description="Title of the source paper, if known.")
    url: str | None = Field(None, description="Link to the source paper, if known.")
    text: str = Field(..., description="The chunk text (or abstract fallback) that constitutes the evidence.")
    paper_id: str | None = Field(None, description="The stored paper_id this chunk belongs to, used for source attribution.")
    score: float | None = Field(None, description="Retrieval relevance (reranker) score, if available.")
