from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from src.core.config import settings
from src.db.models.base import Base

class Paper(Base):
    __tablename__ = "papers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    external_id = Column(String, nullable=False)
    url = Column(String, nullable=True)
    abstract = Column(String, nullable=True)
    authors = Column(ARRAY(String))
    research_id = Column(UUID(as_uuid=True), ForeignKey("researchs.id"), nullable=True)
    published_at = Column(DateTime(timezone=True))
    ingested_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    chunks = relationship(
        "PaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )

class PaperChunk(Base):
    __tablename__ = "paper_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    section = Column(String, nullable=True)
    embedding = Column(Vector(settings.embedding_dim), nullable=False)
    
    paper = relationship("Paper", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_chunk_paper_index"),
    )