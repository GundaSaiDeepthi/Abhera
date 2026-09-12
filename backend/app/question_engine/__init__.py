"""
Dynamic Question Engine Package.

Provides rule-based, state-aware dynamic question selection for incident submissions.
"""

from app.question_engine.engine import process_submission, record_answer
from app.question_engine.rules import get_required_information, normalize_category
from app.question_engine.question_selector import (
    get_known_information,
    get_missing_information,
    select_next_question,
)

__all__ = [
    "process_submission",
    "record_answer",
    "get_required_information",
    "get_known_information",
    "get_missing_information",
    "select_next_question",
    "normalize_category",
]
