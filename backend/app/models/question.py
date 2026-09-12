from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Question(Base):
    """
    Master bank of dynamic questions for the incident questionnaire engine.
    """
    __tablename__ = "questions"

    question_id = Column(String(64), primary_key=True, index=True)
    category = Column(String(100), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    required_entity = Column(String(100), nullable=False, index=True)
    priority = Column(Integer, default=1, nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    # Relationships
    responses = relationship("SubmissionQuestionAnswer", back_populates="question", cascade="all, delete-orphan")


class SubmissionQuestionAnswer(Base):
    """
    Tracks question-answer conversation state for each incident submission.
    """
    __tablename__ = "submission_question_answers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        ForeignKey("incident_submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id = Column(
        String(64),
        ForeignKey("questions.question_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    answer = Column(Text, nullable=True)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    asked_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    submission = relationship("IncidentSubmission", back_populates="question_answers")
    question = relationship("Question", back_populates="responses")
