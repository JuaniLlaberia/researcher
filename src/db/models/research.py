import enum
from uuid import uuid4

from sqlalchemy import Column, Enum, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.db.models.base import Base

class ResearchStatus(enum.Enum):
    PAUSED = "paused"
    ACTIVE = "active"
    COMPLETED = "completed"

class Research(Base):
    __tablename__ = "researchs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    goal = Column(String, nullable=False)
    status = Column(Enum(ResearchStatus))
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
