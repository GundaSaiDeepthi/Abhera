"""
Question Selector Module for Dynamic Question Engine.

Handles known information extraction, missing information computation,
and deterministic, priority-based question selection.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from app.question_engine.rules import normalize_category

logger = logging.getLogger("QuestionSelector")


def get_known_information(
    narrative_text: str = "",
    ner_entities: Any = None,
    previous_answers: Dict[str, Any] = None,
    question_bank_dict: Dict[str, Any] = None,
) -> Dict[str, str]:
    """
    Determines information already known from initial narrative text,
    extracted NER entities, and previously answered questions.

    Returns:
        Dict mapping entity dimension (e.g., 'PERP_REL') to known value/answer string.
    """
    known: Dict[str, str] = {}
    import re

    # 1. Process NER Extracted Entities (handles list of dicts OR dict of lists/strings)
    if ner_entities:
        if isinstance(ner_entities, list):
            for ent in ner_entities:
                if isinstance(ent, dict):
                    label = ent.get("label") or ent.get("entity_type")
                    text = ent.get("text") or ent.get("entity_text")
                    if label and text and label not in known:
                        known[label] = str(text).strip()
        elif isinstance(ner_entities, dict):
            for lbl, vals in ner_entities.items():
                if vals:
                    if isinstance(vals, list) and len(vals) > 0:
                        known[lbl] = str(vals[0]).strip()
                    elif isinstance(vals, str):
                        known[lbl] = vals.strip()

    # 2. Process Previously Answered Questions
    if previous_answers:
        for q_id_or_entity, ans in previous_answers.items():
            if ans and str(ans).strip():
                # Check if key is a question_id mapped to required_entity
                if question_bank_dict and q_id_or_entity in question_bank_dict:
                    req_ent = question_bank_dict[q_id_or_entity].required_entity
                    known[req_ent] = str(ans).strip()
                elif isinstance(q_id_or_entity, str) and q_id_or_entity.isupper():
                    known[q_id_or_entity] = str(ans).strip()

    # 3. Comprehensive Narrative Text Heuristics for Known Dimensions
    if narrative_text and isinstance(narrative_text, str) and narrative_text.strip():
        text = narrative_text.strip()
        lower = text.lower()

        # A. Incident Details: Substantial narrative (>= 4 words or clear incident action)
        if "INCIDENT_DETAILS" not in known and len(text.split()) >= 4:
            known["INCIDENT_DETAILS"] = text

        # B. Safety Status Heuristics
        if "SAFETY_STATUS" not in known:
            safety_patterns = [
                r"\b(currently safe|am safe|feel safe|now safe|safe now|safe place|out of danger|is safe|are safe)\b",
                r"\b(in danger|feel unsafe|not safe|immediate danger|physical danger|danger|threatened|afraid|scared|fear|harm|threats|threatening)\b"
            ]
            for pat in safety_patterns:
                m = re.search(pat, lower)
                if m:
                    known["SAFETY_STATUS"] = m.group(0)
                    break

        # C. Time / Frequency Heuristics
        if "TIME_FREQ" not in known:
            time_patterns = [
                r"\b(every day|daily|every night|nightly|weekly|last week|this week|past week|last month|past month|yesterday|this morning|today|for months|for years|for weeks|since|repeatedly|keeps|kept|always|constantly|frequently|often|ago)\b"
            ]
            for pat in time_patterns:
                m = re.search(pat, lower)
                if m:
                    known["TIME_FREQ"] = m.group(0)
                    break

        # D. Platform Heuristics
        if "PLATFORM" not in known:
            platform_patterns = [
                r"\b(instagram|whatsapp|facebook|twitter|telegram|snapchat|linkedin|email|phone|call|calls|sms|text|texts|message|messages|app|online|website|chat|social media)\b"
            ]
            for pat in platform_patterns:
                m = re.search(pat, lower)
                if m:
                    known["PLATFORM"] = m.group(0)
                    break

        # E. Location Heuristics
        if "LOCATION" not in known:
            location_patterns = [
                r"\b(workplace|office|job|work|college|university|school|class|home|house|apartment|flat|room|street|road|bus|metro|train|station|cab|auto|public|park|store|shop)\b"
            ]
            for pat in location_patterns:
                m = re.search(pat, lower)
                if m:
                    known["LOCATION"] = m.group(0)
                    break

        # F. Perpetrator / Relationship Heuristics
        if "PERP_REL" not in known:
            perp_patterns = [
                r"\b(husband|ex-husband|wife|ex-wife|boyfriend|ex-boyfriend|girlfriend|ex-girlfriend|fiance|ex-fiance|partner|ex-partner|in-law|in-laws|father-in-law|mother-in-law|colleague|coworker|boss|manager|supervisor|employer|landlord|neighbor|stranger|someone|relative|uncle|cousin|father|mother|brother|friend|acquaintance)\b",
                r"\b([A-Z0-9_]+)\s+(?:falsely|promised|follows|stalked|hit|beat)\b"
            ]
            for pat in perp_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    if m.groups() and m.group(1):
                        known["PERP_REL"] = m.group(1)
                    else:
                        known["PERP_REL"] = m.group(0)
                    break

    return known


def get_missing_information(
    required_info: List[str],
    known_info: Dict[str, str],
) -> List[str]:
    """
    Computes missing information dimensions by subtracting known information from required.
    Preserves exact priority order of required information.
    """
    missing = []
    for req in required_info:
        if req not in known_info or not known_info[req]:
            missing.append(req)
    return missing


def select_next_question(
    missing_info: List[str],
    active_categories: List[str],
    asked_question_ids: Set[str],
    questions_bank: List[Any],
) -> Optional[Any]:
    """
    Selects the highest-priority, relevant unanswered question from the bank.

    Selection Rules:
    1. Excludes questions already asked (in asked_question_ids).
    2. Must be active (active == True).
    3. Must target an entity dimension currently in missing_info.
    4. MUST match active categories or General (skips off-category questions).
    5. Highest priority integer score wins.
    """
    if not missing_info or not questions_bank:
        return None

    norm_categories = [normalize_category(c) for c in active_categories] if active_categories else ["General"]
    missing_set = set(missing_info)
    asked_set = set(asked_question_ids) if asked_question_ids else set()

    candidates = []

    for q in questions_bank:
        q_id = getattr(q, "question_id", None)
        q_active = getattr(q, "active", True)
        q_entity = getattr(q, "required_entity", None)
        q_category = normalize_category(getattr(q, "category", "General"))
        q_priority = getattr(q, "priority", 1)

        # 1. Skip if already asked
        if q_id in asked_set:
            continue

        # 2. Skip if inactive
        if not q_active:
            continue

        # 3. Must target a missing entity requirement
        if q_entity not in missing_set:
            continue

        # 4. Skip questions from unrelated specific categories
        if q_category not in norm_categories and q_category != "General":
            continue

        # Calculate relevance score:
        category_relevance = 50 if q_category in norm_categories else 20

        # Priority boost if targeting the first missing requirement
        missing_rank_score = (len(missing_info) - missing_info.index(q_entity)) * 10

        total_score = q_priority + category_relevance + missing_rank_score

        candidates.append((total_score, q_priority, q))

    if not candidates:
        return None

    # Sort candidates by total_score descending
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]
