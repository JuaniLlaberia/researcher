import enum
from uuid import uuid4

from sqlalchemy import Column, Enum, String, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.db.models.base import Base

class Stance(enum.Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"

class Finding(Base):
    __tablename__ = "findings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    content = Column(String, nullable=False)
    stance = Column(Enum(Stance))
    research_id = Column(UUID(as_uuid=True), ForeignKey("researchs.id"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    hypothesis_id = Column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())