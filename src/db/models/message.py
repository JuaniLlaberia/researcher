import enum
from uuid import uuid4

from sqlalchemy import Column, Enum, DateTime, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.db.models.base import Base

class Role(enum.Enum):
    USER = "user"
    AGENT = "agent"

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    role = Column(Enum(Role), nullable=False)
    content = Column(String, nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    research_id = Column(UUID(as_uuid=True), ForeignKey("researchs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())