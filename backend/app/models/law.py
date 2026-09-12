from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func
from app.database import Base


class Law(Base):
    __tablename__ = "laws"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    act_name = Column(String(100), nullable=False, index=True)
    section_number = Column(String(50), nullable=False)
    section_text = Column(Text, nullable=False)
    applicable_label = Column(String(50), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
