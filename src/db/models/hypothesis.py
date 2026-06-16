import enum
from uuid import uuid4

from sqlalchemy import CheckConstraint, Column, Enum, Float, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from src.db.models.base import Base

class HypothesisStatus(enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    PARKED = "parked"

class Hypothesis(Base):
    __tablename__ = "hypotheses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    text = Column(String, nullable=False)
    score = Column(Float())
    status = Column(Enum(HypothesisStatus), nullable=False)
    research_id = Column(UUID(as_uuid=True), ForeignKey("researchs.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), nullable=True) # Which hypothesis this one was derived from

    __table_args__ = (
        CheckConstraint("score >= 0.0 AND score <= 1.0", name="check_score_range"),
    )
