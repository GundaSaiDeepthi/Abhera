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
