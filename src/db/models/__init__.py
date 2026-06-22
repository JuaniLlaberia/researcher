"""
Model package

IMPORTANT: every model module MUST be imported here. SQLAlchemy only registers a
table on `Base.metadata` once its module has been imported, so any model not
imported here is invisible to Alembic autogenerate / create_all, and string
relationships (e.g. relationship("Paper")) will fail to resolve.
"""
from src.db.models.base import Base
from src.db.models.hypothesis import Hypothesis
from src.db.models.research import Research
from src.db.models.paper import Paper, PaperChunk
from src.db.models.session import Session
from src.db.models.finding import Finding
from src.db.models.message import Message

__all__ = [
    "Base",
    "Hypothesis",
    "Research",
    "Paper",
    "PaperChunk",
    "Session",
    "Finding",
    "Message"
]
