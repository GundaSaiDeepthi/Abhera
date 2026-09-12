import uuid
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class IncidentSubmission(Base):
    __tablename__ = "incident_submissions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: f"SUB-{str(uuid.uuid4())[:8]}",
    )
    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    narrative_text = Column(Text, nullable=False)
    state = Column(String(100), nullable=True, index=True)
    district = Column(String(100), nullable=True, index=True)
    predicted_labels = Column(JSON, nullable=True)
    extracted_entities = Column(JSON, nullable=True)
    report_generated = Column(Boolean, default=False, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session = relationship("Session", back_populates="submissions")
    messages = relationship(
        "ConversationMessage",
        back_populates="submission",
        cascade="all, delete-orphan",
    )
    predictions = relationship(
        "Prediction",
        back_populates="submission",
        cascade="all, delete-orphan",
    )
    entities = relationship(
        "Entity",
        back_populates="submission",
        cascade="all, delete-orphan",
    )
    question_answers = relationship(
        "SubmissionQuestionAnswer",
        back_populates="submission",
        cascade="all, delete-orphan",
    )
    report = relationship(
        "Report",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan",
    )
