"""
ABHERA — Central Conversational Chat Service Orchestrator

This service encapsulates the core business logic for processing user messages
throughout the multi-turn conversational intake workflow.

Pipeline Flow & Orchestration Responsibilities:
----------------------------------------------
1. Session & Submission Lifecycle: Fetches or creates DB session and incident submission records.
2. Questionnaire Response Capture: Records user answer if currently answering a dynamic follow-up question.
3. ML Prediction Execution:
   - Invokes fine-tuned BERT Multi-Label Classifier (`predict_bert.py`) to categorize legal incident classes.
   - Invokes fine-tuned NER Entity Extraction (`predict_ner.py`) to extract evidentiary entities (PERP_REL, LOCATION, etc.).
4. Narrative Consolidation: Merges initial survivor narrative with answer context for complete incident representation.
5. Dynamic Question Engine: Selects contextually relevant, non-duplicate follow-up questions to fill missing information gaps.
6. Legal & Support Knowledge-Base Retrieval: Maps verified statutory laws (IPC/BNS) and support helplines from DB.
7. Anti-Hallucination & Provenance Safeguard: Ensures responses derive strictly from verified DB content.
8. Report Generation & DB Persistence: Triggers automatic Markdown report compilation and updates session state.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm.attributes import flag_modified

# Ensure root project directory is in sys.path so ml module can be imported
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.models.session import Session as UserSession
from app.models.incident_submission import IncidentSubmission
from app.services.session_service import SessionService
from app.question_engine.engine import process_submission, record_answer
from app.services.report_generator import generate_incident_report
from app.services.input_validator import (
    validate_input,
    VALID_INCIDENT,
    INSUFFICIENT_INFORMATION,
    UNCERTAIN_CLASSIFICATION,
    NON_INCIDENT_STATUSES,
)

# Import ML Predictors lazily to handle environments cleanly
try:
    from ml.inference.predict_bert import BERTIncidentPredictor
except Exception as _e:
    logging.warning(f"Could not import BERTIncidentPredictor: {_e}")
    BERTIncidentPredictor = None

try:
    from ml.inference.predict_ner import NEREntityPredictor
except Exception as _e:
    logging.warning(f"Could not import NEREntityPredictor: {_e}")
    NEREntityPredictor = None

logger = logging.getLogger("ChatService")


class ChatService:
    _bert_predictor: Optional[Any] = None
    _ner_predictor: Optional[Any] = None

    @classmethod
    def get_bert_predictor(cls):
        if cls._bert_predictor is None and BERTIncidentPredictor is not None:
            try:
                cls._bert_predictor = BERTIncidentPredictor()
            except Exception as e:
                logger.warning(f"Could not initialize BERTIncidentPredictor: {e}")
        return cls._bert_predictor

    @classmethod
    def get_ner_predictor(cls):
        if cls._ner_predictor is None and NEREntityPredictor is not None:
            try:
                cls._ner_predictor = NEREntityPredictor()
            except Exception as e:
                logger.warning(f"Could not initialize NEREntityPredictor: {e}")
        return cls._ner_predictor

    @staticmethod
    def process_message(
        db: DBSession,
        session_id: str,
        user_message_text: str,
    ) -> Dict[str, Any]:
        """
        Orchestrates session lookup, user message saving, BERT classification,
        NER extraction, Dynamic Question Engine execution, Legal/Support mapping,
        Anti-Hallucination validation, and Report Generation.
        """
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id cannot be empty")

        if not user_message_text or not str(user_message_text).strip():
            raise ValueError("message text cannot be empty")

        clean_text = user_message_text.strip()
        session = SessionService.get_session(db, session_id.strip())
        if not session:
            raise KeyError(f"Session with ID '{session_id}' not found")

        messages_list = list(session.messages or [])
        prior_user_msgs = [
            m for m in messages_list if m.get("role") == "user" or m.get("sender") == "user"
        ]

        # 0. Input Relevance Validation (runs BEFORE BERT / NER / Question Engine / Report).
        # Once an incident has been accepted, later messages are answers to follow-up
        # questions and are not re-validated (short answers like "Delhi" are legitimate).
        # Legacy messages without a relevance tag were processed by the original pipeline.
        incident_established = bool(session.current_question) or any(
            m.get("relevance_status", VALID_INCIDENT) == VALID_INCIDENT for m in prior_user_msgs
        )

        validation: Dict[str, Any]
        incident_narrative: Optional[str] = None
        if incident_established:
            validation = {"status": VALID_INCIDENT, "reason": "follow-up message for an accepted incident"}
        else:
            # Combine earlier clarification-stage messages with the current one so that a
            # user who first wrote "I am facing a problem." can complete the description.
            pending = [
                (m.get("content") or m.get("text") or "")
                for m in prior_user_msgs
                if m.get("relevance_status") in (INSUFFICIENT_INFORMATION, UNCERTAIN_CLASSIFICATION)
            ]
            candidate = " ".join([t for t in pending if t] + [clean_text])
            bert_for_validation = ChatService.get_bert_predictor()
            validation = validate_input(
                candidate,
                bert_predict=bert_for_validation.predict if bert_for_validation else None,
            )
            if validation["status"] == VALID_INCIDENT:
                incident_narrative = candidate

        input_status = validation["status"]

        if input_status != VALID_INCIDENT:
            return ChatService._respond_without_analysis(db, session, messages_list, clean_text, validation)

        # Retrieve or create incident submission record (only for accepted incidents)
        submission = (
            db.query(IncidentSubmission)
            .filter(IncidentSubmission.submission_id == session.submission_id)
            .first()
        )
        if not submission:
            submission = IncidentSubmission(
                submission_id=session.submission_id,
                session_id=session.session_id,
                narrative_text=incident_narrative or clean_text,
                predicted_labels=[],
                extracted_entities=[],
            )
            db.add(submission)
            db.commit()

        # Record answer if user is answering a previous question
        if session.current_question and isinstance(session.current_question, dict):
            q_id = session.current_question.get("question_id")
            if q_id:
                record_answer(session.submission_id, q_id, clean_text, db)
                answers_list = list(session.answers or [])
                answers_list.append({"question_id": q_id, "answer": clean_text})
                session.answers = answers_list
                flag_modified(session, "answers")

        # Save user message in session
        now_str = datetime.now(timezone.utc).strftime("%H:%M")
        user_msg_obj = {
            "id": f"msg-{int(datetime.now(timezone.utc).timestamp()*1000)}",
            "role": "user",
            "sender": "user",
            "content": clean_text,
            "text": clean_text,
            "timestamp": now_str,
            "relevance_status": input_status,
        }

        messages_list.append(user_msg_obj)
        session.messages = messages_list
        flag_modified(session, "messages")

        # Build full narrative for ML evaluation from incident-relevant user messages only
        # (greetings / out-of-scope / invalid messages are excluded).
        user_messages_texts = [
            m.get("content") or m.get("text")
            for m in messages_list
            if (m.get("role") == "user" or m.get("sender") == "user")
            and m.get("relevance_status", VALID_INCIDENT) not in NON_INCIDENT_STATUSES
        ]
        full_narrative = "\n".join([t for t in user_messages_texts if t])

        # Initial incident description: the text accepted by the relevance validator
        # (falls back to the first relevant user message for legacy sessions).
        if incident_narrative:
            initial_narrative = incident_narrative
        elif submission.narrative_text:
            initial_narrative = submission.narrative_text
        else:
            initial_narrative = user_messages_texts[0] if user_messages_texts else clean_text
        submission.narrative_text = initial_narrative

        # 1. BERT Classifier Integration
        predicted_labels = list(session.predicted_labels or [])
        bert_predictor = ChatService.get_bert_predictor()
        if bert_predictor:
            try:
                bert_res = bert_predictor.predict(full_narrative)
                labels_objs = bert_res.get("predicted_labels", [])
                new_labels = [l["name"] for l in labels_objs]
                if new_labels:
                    for nl in new_labels:
                        if nl not in predicted_labels:
                            predicted_labels.append(nl)
                elif not predicted_labels and initial_narrative:
                    # Check initial incident narrative if short follow-up answers diluted the full narrative
                    initial_text = initial_narrative
                    init_res = bert_predictor.predict(initial_text)
                    init_objs = init_res.get("predicted_labels", [])
                    for il in [l["name"] for l in init_objs]:
                        if il not in predicted_labels:
                            predicted_labels.append(il)
            except Exception as e:
                logger.warning(f"BERT prediction error: {e}")

        session.predicted_labels = predicted_labels
        flag_modified(session, "predicted_labels")
        submission.predicted_labels = {"labels": predicted_labels}
        flag_modified(submission, "predicted_labels")

        # 2. NER Entity Extraction Integration
        existing_entities_dict = dict(session.entities or {})
        existing_entities_list = list(submission.extracted_entities or [])
        ner_predictor = ChatService.get_ner_predictor()

        if ner_predictor:
            try:
                extracted = ner_predictor.predict(clean_text)
                for ent in extracted:
                    lbl = ent["label"]
                    txt = ent["text"]
                    if lbl not in existing_entities_dict:
                        existing_entities_dict[lbl] = []
                    if txt not in existing_entities_dict[lbl]:
                        existing_entities_dict[lbl].append(txt)

                    if not any(e.get("label") == lbl and e.get("text") == txt for e in existing_entities_list):
                        existing_entities_list.append({"label": lbl, "text": txt})
            except Exception as e:
                logger.warning(f"NER extraction error: {e}")

        session.entities = existing_entities_dict
        flag_modified(session, "entities")
        submission.extracted_entities = existing_entities_list
        flag_modified(submission, "extracted_entities")
        db.commit()

        # 3. Dynamic Question Engine Integration
        question_response = process_submission(
            submission_id=session.submission_id,
            narrative_text=initial_narrative,
            bert_results=predicted_labels,
            ner_entities=existing_entities_list,
            db=db,
        )

        report_obj = None

        if not question_response.get("completion_status") and question_response.get("next_question"):
            # Follow-up question required
            session.current_question = {
                "question_id": question_response.get("question_id"),
                "question_text": question_response.get("next_question"),
                "missing_information": question_response.get("missing_information", []),
            }
            session.conversation_status = "QUESTIONING"
            assistant_text = question_response["next_question"]
        else:
            # Questioning complete -> Processing & Report Generation
            session.current_question = None
            session.conversation_status = "PROCESSING"

            # Execute Legal Mapping, Support Mapping, Anti-Hallucination & Report Generator
            report_obj = generate_incident_report(
                submission_id=session.submission_id,
                narrative_text=initial_narrative,
                bert_results=predicted_labels,
                ner_entities=existing_entities_list,
                user_answers=session.answers,
                state=submission.state,
                district=submission.district,
                db=db,
            )

            session.conversation_status = "COMPLETED"
            assistant_text = (
                "Thank you. All required information has been collected and your structured incident report has been generated."
            )

        # Append assistant message to session
        assistant_msg_obj = {
            "id": f"msg-{int(datetime.now(timezone.utc).timestamp()*1000)+1}",
            "role": "assistant",
            "sender": "assistant",
            "content": assistant_text,
            "text": assistant_text,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
        }
        messages_list.append(assistant_msg_obj)
        session.messages = messages_list
        flag_modified(session, "messages")

        db.commit()
        db.refresh(session)

        return {
            "session_id": session.session_id,
            "submission_id": session.submission_id,
            "assistant_message": assistant_text,
            "predicted_labels": session.predicted_labels or [],
            "entities": session.entities or {},
            "current_question": session.current_question,
            "conversation_status": session.conversation_status,
            "report": report_obj,
            "input_status": input_status,
            "validation_reason": validation.get("reason"),
        }

    @staticmethod
    def _respond_without_analysis(
        db: DBSession,
        session: UserSession,
        messages_list: List[Dict[str, Any]],
        clean_text: str,
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handles OUT_OF_SCOPE, INVALID_INPUT, INSUFFICIENT_INFORMATION and
        UNCERTAIN_CLASSIFICATION: stores the exchange but runs NO BERT labelling,
        NER, question engine, legal/support retrieval or report generation, and
        does not create an incident submission record.
        """
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        now_str = datetime.now(timezone.utc).strftime("%H:%M")
        assistant_text = validation["message"]
        messages_list.append({
            "id": f"msg-{ts}", "role": "user", "sender": "user",
            "content": clean_text, "text": clean_text, "timestamp": now_str,
            "relevance_status": validation["status"],
        })
        messages_list.append({
            "id": f"msg-{ts + 1}", "role": "assistant", "sender": "assistant",
            "content": assistant_text, "text": assistant_text, "timestamp": now_str,
        })
        session.messages = messages_list
        flag_modified(session, "messages")
        session.current_question = None
        session.conversation_status = "ACTIVE"
        db.commit()
        db.refresh(session)

        return {
            "session_id": session.session_id,
            "submission_id": session.submission_id,
            "assistant_message": assistant_text,
            "predicted_labels": [],
            "entities": {},
            "current_question": None,
            "conversation_status": session.conversation_status,
            "report": None,
            "input_status": validation["status"],
            "validation_reason": validation.get("reason"),
        }
