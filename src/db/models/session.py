from uuid import uuid4

from sqlalchemy import Column, DateTime, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.db.models.base import Base

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    research_id = Column(UUID(as_uuid=True), ForeignKey("researchs.id"), nullable=False)
    summary = Column(String, nullable=True)
    hitl_mode = Column(String, nullable=False, default="checkpoint")
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())