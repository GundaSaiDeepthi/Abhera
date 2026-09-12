from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func
from app.database import Base


class SupportService(Base):
    __tablename__ = "support_services"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    service_type = Column(String(50), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    contact_number = Column(String(255), nullable=False)
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)
    applicable_label = Column(String(255), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
