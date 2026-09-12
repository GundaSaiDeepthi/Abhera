from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SessionResponse(BaseModel):
    session_id: str
    submission_id: Optional[str] = None
    messages: List[Any] = Field(default_factory=list)
    answers: List[Any] = Field(default_factory=list)
    predicted_labels: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    current_question: Optional[Dict[str, Any]] = None
    conversation_status: str = "ACTIVE"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionStartRequest(BaseModel):
    user_id: Optional[int] = None
    initial_narrative: Optional[str] = ""

    model_config = ConfigDict(from_attributes=True)


class SessionUpdateRequest(BaseModel):
    messages: Optional[List[Any]] = None
    answers: Optional[List[Any]] = None
    predicted_labels: Optional[List[str]] = None
    entities: Optional[Dict[str, Any]] = None
    current_question: Optional[Dict[str, Any]] = None
    conversation_status: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
