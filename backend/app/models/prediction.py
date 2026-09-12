from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        ForeignKey("incident_submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    is_positive = Column(Boolean, default=True, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship
    submission = relationship("IncidentSubmission", back_populates="predictions")
