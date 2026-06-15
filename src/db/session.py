"""
SQLAlchemy engines and session factories.
Two engines share one configured database, sourced from `core.config.settings`:

* sync (`engine` / `SessionLocal`): Alembic migrations and standalone scripts.
* async (`async_engine` / `AsyncSessionLocal`): the default for app runtime
  (graph nodes, FastAPI), matching the async direction of the codebase.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

# --- Sync (migrations, scripts) ---
engine = create_engine(settings.database_url_sync, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

# --- Async (app runtime) ---
async_engine = create_async_engine(
    settings.database_url_async, pool_pre_ping=True, future=True
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)
