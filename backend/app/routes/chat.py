"""
ABHERA — Conversational Chat API Router

This router exposes the primary `/api/chat/message` REST API endpoint used by
the ABHERA web frontend to submit survivor narrative statements and questionnaire responses.

End-to-End Execution Flow:
--------------------------
1. Request Validation: Ensures valid non-empty `session_id` and `message` payload.
2. Service Delegation: Hands off execution to `ChatService.process_message()`.
3. Multi-Module Processing:
   - Persists user message to DB.
   - Runs fine-tuned BERT multi-label classification (`predict_bert.py`).
   - Runs fine-tuned NER entity extraction (`predict_ner.py`).
   - Evaluates Dynamic Question Engine state.
   - Maps verified Legal (IPC/BNS) and Support Helpline information from DB.
4. Response Serialization: Returns structured `ChatMessageResponse` containing
   bot reply text, extracted entities, dynamic questions, and progress metadata.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from app.database import get_db
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["Conversational Chat"])


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Process conversational incident message",
    description="Processes a user message through BERT classification, NER entity extraction, Dynamic Question Engine, and Legal/Support mapping.",
)
def send_chat_message(
    payload: ChatMessageRequest,
    db: DBSession = Depends(get_db),
):
    """
    Handles user chat message submission for an active session.
    """
    if not payload.session_id or not payload.session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty.",
        )

    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    try:
        response_data = ChatService.process_message(
            db=db,
            session_id=payload.session_id.strip(),
            user_message_text=payload.message.strip(),
        )
        return response_data
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the message: {str(e)}",
        )
