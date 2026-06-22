"""
Create the application tables in the (already running) Postgres database.

This does NOT create the Postgres instance or the pgvector extension — Docker and
init/01-extension.sql handle those. It only creates the ORM tables, and is
idempotent: `create_all` skips tables that already exist, so re-running is safe.

Run from anywhere:  python scripts/init_db.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect  # noqa: E402

import src.db.models  # noqa: F401,E402  — imports every model so metadata is complete
from src.db.models import Base  # noqa: E402
from src.db.session import engine  # noqa: E402

def main() -> int:
    Base.metadata.create_all(engine)
    tables = inspect(engine).get_table_names()
    print(f"✓ schema ready — tables: {sorted(tables)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
