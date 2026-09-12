from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from app.database import get_db
from app.schemas.session import SessionResponse, SessionStartRequest
from app.services.session_service import SessionService

router = APIRouter(prefix="/api/session", tags=["Session Management"])


@router.post(
    "/start",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new conversation session",
    description="Initializes a new user conversation session and associates an incident submission record.",
)
def start_session(
    payload: SessionStartRequest = SessionStartRequest(),
    db: DBSession = Depends(get_db),
):
    """
    Creates and initializes a new conversation session.
    """
    try:
        session = SessionService.create_session(
            db,
            user_id=payload.user_id,
            initial_narrative=payload.initial_narrative or "",
        )
        return session
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start conversation session: {str(e)}",
        )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve session state",
    description="Fetches an existing conversation session state by session_id.",
)
def get_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """
    Retrieves the current state of an existing conversation session.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty.",
        )

    session = SessionService.get_session(db, session_id.strip())
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )

    return session
