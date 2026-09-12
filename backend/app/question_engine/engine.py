"""
Dynamic Question Engine Main Orchestrator.

Combines rules, question selection, and state tracking to dynamically adapt
the conversation flow for incident submissions.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timezone

from app.models.question import Question, SubmissionQuestionAnswer
from app.question_engine.rules import get_required_information, normalize_category
from app.question_engine.question_selector import (
    get_known_information,
    get_missing_information,
    select_next_question,
)
from app.question_engine.dynamic_generator import generate_dynamic_question

logger = logging.getLogger("DynamicQuestionEngine")


def record_answer(
    submission_id: str,
    question_id: str,
    answer_text: str,
    db: DBSession,
) -> SubmissionQuestionAnswer:
    """
    Records or updates a user's answer to a specific question for a submission.
    """
    existing = (
        db.query(SubmissionQuestionAnswer)
        .filter(
            SubmissionQuestionAnswer.submission_id == submission_id,
            SubmissionQuestionAnswer.question_id == question_id,
        )
        .first()
    )

    now = datetime.now(timezone.utc)
    if existing:
        existing.answer = answer_text
        existing.answered_at = now
        db.commit()
        db.refresh(existing)
        return existing
    else:
        new_record = SubmissionQuestionAnswer(
            submission_id=submission_id,
            question_id=question_id,
            answer=answer_text,
            answered_at=now,
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        return new_record


def process_submission(
    submission_id: str,
    narrative_text: str = "",
    bert_results: Dict[str, Any] = None,
    ner_entities: List[Dict[str, Any]] = None,
    previous_answers: Dict[str, str] = None,
    db: Optional[DBSession] = None,
    questions_bank_override: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Processes an incident submission through the Dynamic Question Engine.

    Args:
        submission_id: Unique submission ID
        narrative_text: Initial incident narrative
        bert_results: Dict containing BERT predicted labels/scores (e.g. {'labels': ['DV', 'ST']})
        ner_entities: List of extracted NER entity dicts
        previous_answers: Dict of previous answers (used when db is None or as fallback)
        db: Optional SQLAlchemy Session for PostgreSQL persistence
        questions_bank_override: Optional list of Question objects (for testing without DB)

    Returns:
        Engine response dict matching specification:
        {
            "submission_id": "...",
            "missing_information": [...],
            "next_question": "..." or null,
            "question_id": "..." or null,
            "completion_status": bool
        }
    """
    # 1. Extract Active Categories from BERT Results
    active_categories: List[str] = []
    if bert_results:
        if isinstance(bert_results, dict):
            raw_labels = bert_results.get("labels") or bert_results.get("predicted_labels") or []
            if isinstance(raw_labels, list):
                active_categories = [normalize_category(lbl) for lbl in raw_labels]
        elif isinstance(bert_results, list):
            active_categories = [normalize_category(lbl) for lbl in bert_results]

    if not active_categories:
        active_categories = ["General"]

    # 2. Retrieve Questions Bank
    if questions_bank_override is not None:
        questions_bank = questions_bank_override
    elif db is not None:
        questions_bank = db.query(Question).filter(Question.active == True).all()
    else:
        questions_bank = []

    question_bank_dict = {
        getattr(q, "question_id", str(idx)): q for idx, q in enumerate(questions_bank)
    }

    # 3. Retrieve DB Conversation State (if db session provided)
    asked_question_ids: Set[str] = set()
    answers_dict: Dict[str, str] = previous_answers.copy() if previous_answers else {}

    if db is not None:
        db_records = (
            db.query(SubmissionQuestionAnswer)
            .filter(SubmissionQuestionAnswer.submission_id == submission_id)
            .all()
        )
        for rec in db_records:
            asked_question_ids.add(rec.question_id)
            if rec.answer:
                answers_dict[rec.question_id] = rec.answer
    elif previous_answers:
        for q_id in previous_answers.keys():
            asked_question_ids.add(q_id)

    # 4. Determine Required, Known, and Missing Information
    required_info = get_required_information(active_categories)
    known_info = get_known_information(
        narrative_text=narrative_text,
        ner_entities=ner_entities,
        previous_answers=answers_dict,
        question_bank_dict=question_bank_dict,
    )
    missing_info = get_missing_information(required_info, known_info)

    # 5. Select Next Question
    next_q = select_next_question(
        missing_info=missing_info,
        active_categories=active_categories,
        asked_question_ids=asked_question_ids,
        questions_bank=questions_bank,
    )

    # 6. Build Final Engine Response
    if not missing_info or next_q is None:
        return {
            "submission_id": submission_id,
            "missing_information": [],
            "next_question": None,
            "question_id": None,
            "completion_status": True,
        }

    q_id = getattr(next_q, "question_id", None)
    q_entity = getattr(next_q, "required_entity", None) or (missing_info[0] if missing_info else "GENERAL")
    fallback_q_text = getattr(next_q, "question_text", None)

    dynamic_q_text = generate_dynamic_question(
        required_entity=q_entity,
        active_categories=active_categories,
        narrative_text=narrative_text,
        known_info=known_info,
        fallback_text=fallback_q_text,
    )

    return {
        "submission_id": submission_id,
        "missing_information": missing_info,
        "next_question": dynamic_q_text,
        "question_id": q_id,
        "completion_status": False,
    }
