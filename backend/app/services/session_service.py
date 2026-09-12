import uuid
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session as DBSession
from app.models.session import Session as UserSession, ALLOWED_CONVERSATION_STATUSES
from app.models.incident_submission import IncidentSubmission


class SessionService:

    @staticmethod
    def create_session(
        db: DBSession,
        user_id: Optional[int] = None,
        initial_narrative: str = "",
    ) -> UserSession:
        """
        Creates a new conversation session, associates an incident submission record,
        initializes default state, and persists to PostgreSQL.
        """
        sess_uuid = str(uuid.uuid4())[:8]
        sess_id = f"SESS-{sess_uuid}"
        sub_id = f"SUB-{sess_uuid}"

        session = UserSession(
            session_id=sess_id,
            submission_id=sub_id,
            user_id=user_id,
            messages=[],
            answers=[],
            predicted_labels=[],
            entities={},
            current_question=None,
            conversation_status="ACTIVE",
        )
        db.add(session)
        db.flush()

        submission = IncidentSubmission(
            submission_id=sub_id,
            session_id=sess_id,
            narrative_text=initial_narrative or "",
            predicted_labels=[],
            extracted_entities=[],
        )
        db.add(submission)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_session(db: DBSession, session_id: str) -> Optional[UserSession]:
        """Retrieves an existing session by session_id."""
        if not session_id or not isinstance(session_id, str):
            return None
        db.expire_all()
        return db.query(UserSession).filter(UserSession.session_id == session_id.strip()).first()

    @staticmethod
    def update_session(
        db: DBSession,
        session_id: str,
        updates: Dict[str, Any],
    ) -> Optional[UserSession]:
        """
        Safely updates an existing session's state in PostgreSQL.
        Validates conversation_status values.
        """
        session = SessionService.get_session(db, session_id)
        if not session:
            return None

        if "conversation_status" in updates and updates["conversation_status"] is not None:
            status = updates["conversation_status"].strip()
            if status not in ALLOWED_CONVERSATION_STATUSES:
                raise ValueError(
                    f"Invalid conversation_status '{status}'. Must be one of: {sorted(list(ALLOWED_CONVERSATION_STATUSES))}"
                )
            session.conversation_status = status

        if "messages" in updates and updates["messages"] is not None:
            session.messages = updates["messages"]

        if "answers" in updates and updates["answers"] is not None:
            session.answers = updates["answers"]

        if "predicted_labels" in updates and updates["predicted_labels"] is not None:
            session.predicted_labels = updates["predicted_labels"]

        if "entities" in updates and updates["entities"] is not None:
            session.entities = updates["entities"]

        if "current_question" in updates:
            session.current_question = updates["current_question"]

        if "submission_id" in updates and updates["submission_id"] is not None:
            session.submission_id = updates["submission_id"]

        db.commit()
        db.refresh(session)
        return session
