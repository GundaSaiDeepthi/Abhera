from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID of the conversation")
    message: str = Field(..., description="User's incident description or follow-up answer")

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    session_id: str
    submission_id: str
    assistant_message: str
    predicted_labels: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    current_question: Optional[Dict[str, Any]] = None
    conversation_status: str
    report: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)
