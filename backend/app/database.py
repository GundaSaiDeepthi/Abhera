"""
ABHERA — Relational Database Management & ORM Initialization

This module manages the SQLAlchemy engine initialization, session factory setup,
and request-scoped database connection dependency injection for FastAPI.

Database Persistence Architecture:
-----------------------------------
1. Engine & Pooling: Configures SQLAlchemy engine using URL derived from `config.py`.
   Includes `pool_pre_ping=True` to prevent stale database connection errors.
2. Declarative Base: Provides `Base` class inherited by all ORM entities (Users, Sessions,
   Submissions, Messages, Predictions, Entities, Questions, Laws, Support Services, Reports).
3. Session Lifecycle: `get_db()` yields a transactional database session per HTTP request,
   guaranteeing deterministic connection cleanup and rollback/commit scoping.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Construct SQLAlchemy database engine
SQLALCHEMY_DATABASE_URL = settings.get_database_url()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)

# Session factory for DB transactions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for SQLAlchemy ORM models
Base = declarative_base()


def get_db() -> Generator:
    """Dependency generator yields database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
