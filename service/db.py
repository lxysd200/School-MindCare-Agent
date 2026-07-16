from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine, URL
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

class Base(DeclarativeBase):
    pass

load_dotenv()

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_database_url() -> str | URL:
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    if database_url:
        return database_url

    host = (os.getenv("MYSQL_HOST") or "").strip()
    port = (os.getenv("MYSQL_PORT") or "3306").strip()
    user = (os.getenv("MYSQL_USER") or "").strip()
    password = (os.getenv("MYSQL_PASSWORD") or "").strip()
    database = (os.getenv("MYSQL_DATABASE") or "").strip()

    missing = [
        name
        for name, value in (
            ("MYSQL_HOST", host),
            ("MYSQL_USER", user),
            ("MYSQL_PASSWORD", password),
            ("MYSQL_DATABASE", database),
        )
        if not value
    ]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(f"MySQL connection config is missing: {missing_fields}")

    return URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
        query={"charset": "utf8mb4"},
    )


def get_engine() -> Engine:
    global _engine

    if _engine is None:
        _engine = create_engine(
            _build_database_url(),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_db_connection() -> Connection:
    return get_engine().connect()


def get_db_session() -> Session:
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionLocal()


__all__ = [
    "Base",
    "get_engine",
    "get_db_connection",
    "get_db_session",
]
