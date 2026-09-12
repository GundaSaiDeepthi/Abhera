from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        ForeignKey("incident_submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type = Column(String(50), nullable=False, index=True)
    entity_value = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship
    submission = relationship("IncidentSubmission", back_populates="entities")
