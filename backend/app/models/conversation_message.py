from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        ForeignKey("incident_submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sender = Column(String(20), nullable=False)  # 'user' or 'assistant'
    message_text = Column(Text, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship
    submission = relationship("IncidentSubmission", back_populates="messages")
