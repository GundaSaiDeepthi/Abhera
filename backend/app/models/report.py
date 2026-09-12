from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(
        String(64),
        ForeignKey("incident_submissions.submission_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    report_data = Column(JSON, nullable=False)
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

    # Relationship
    submission = relationship("IncidentSubmission", back_populates="report")
