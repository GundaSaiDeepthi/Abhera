import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

ALLOWED_CONVERSATION_STATUSES = {"ACTIVE", "QUESTIONING", "PROCESSING", "COMPLETED"}


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: f"SESS-{str(uuid.uuid4())[:8]}",
    )
    submission_id = Column(
        String(64),
        nullable=True,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    messages = Column(JSON, nullable=True, default=list)
    answers = Column(JSON, nullable=True, default=list)
    predicted_labels = Column(JSON, nullable=True, default=list)
    entities = Column(JSON, nullable=True, default=dict)
    current_question = Column(JSON, nullable=True)
    conversation_status = Column(
        String(30),
        nullable=False,
        default="ACTIVE",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="sessions")
    submissions = relationship(
        "IncidentSubmission",
        back_populates="session",
        cascade="all, delete-orphan",
    )
