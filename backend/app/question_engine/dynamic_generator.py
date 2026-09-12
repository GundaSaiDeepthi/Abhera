"""
Dynamic Question Generator Module.

Constructs context-aware, personalized follow-up questions tailored to the user's
incident narrative and missing required entity dimensions (CONTROLLED IN PURPOSE, DYNAMIC IN CONTENT).
"""

import re
from typing import Any, Dict, List, Optional
from app.question_engine.rules import normalize_category


def extract_incident_context_phrase(
    narrative_text: str = "",
    known_info: Dict[str, str] = None,
) -> Optional[str]:
    """
    Extracts a short, natural context summary phrase from known_info or narrative_text
    to reference in the generated question (e.g. "your husband beats you every day at home"
    or "someone following you after college").
    """
    known = known_info or {}

    # 1. Extract details or narrative sentence snippet
    details = known.get("INCIDENT_DETAILS")
    perp = known.get("PERP_REL")
    loc = known.get("LOCATION")
    plat = known.get("PLATFORM")

    if details and len(details.strip()) > 0:
        clean_det = details.strip().strip(".!,").lower()
        # Strip trailing safety assertions if present in details
        clean_det = re.sub(
            r"\s+and\s+i\s+(?:am|feel|currently)\s+.*$",
            "",
            clean_det,
            flags=re.IGNORECASE,
        )
        if perp and perp.lower() not in clean_det and not clean_det.startswith("my "):
            return f"the incident involving your {perp} ({clean_det})"
        return clean_det

    if perp and loc:
        return f"the incident involving your {perp} at {loc}"

    if perp:
        return f"the incident involving your {perp}"

    if plat:
        return f"the messages received on {plat}"

    if loc:
        return f"what occurred at {loc}"

    if narrative_text and narrative_text.strip():
        first_line = narrative_text.strip().split("\n")[0]
        snippet = first_line.strip().strip(".!,")
        words = snippet.split()
        if len(words) <= 12:
            return snippet.lower()
        return " ".join(words[:8]).lower() + "..."

    return None


def generate_dynamic_question(
    required_entity: str,
    active_categories: List[str],
    narrative_text: str = "",
    known_info: Dict[str, str] = None,
    fallback_text: Optional[str] = None,
) -> str:
    """
    Generates a dynamic, context-aware question based on the missing entity dimension,
    the user's narrative context, and the active categories.

    Rules:
    - Asks ONLY about the required_entity dimension.
    - References the actual incident narrative when context is available.
    - Does not introduce unconfigured requirements (controlled in purpose).
    - Does not make legal conclusions or assumptions.
    """
    known = known_info or {}
    categories = [normalize_category(c) for c in active_categories] if active_categories else ["General"]
    context_phrase = extract_incident_context_phrase(narrative_text, known)

    # Clean phrase prefix if necessary
    phrase_str = context_phrase
    if phrase_str:
        if phrase_str.startswith("my "):
            phrase_str = "your " + phrase_str[3:]

    # 1. SAFETY_STATUS Dimension
    if required_entity == "SAFETY_STATUS":
        if phrase_str:
            return (
                f"You mentioned that {phrase_str}. "
                f"Are you currently in immediate danger or do you need emergency assistance right now?"
            )
        return (
            "Based on the situation you described, are you currently in immediate physical danger "
            "or in need of emergency assistance right now?"
        )

    # 2. PERP_REL Dimension (Perpetrator / Relationship)
    elif required_entity == "PERP_REL":
        if phrase_str:
            return (
                f"Regarding {phrase_str}, could you clarify your relationship with the person "
                f"involved (e.g., spouse, ex-partner, coworker, relative, or stranger)?"
            )
        return (
            "Could you clarify your relationship with the person involved in this incident "
            "(e.g., spouse, ex-partner, employer/coworker, relative, or stranger)?"
        )

    # 3. TIME_FREQ Dimension (Time / Frequency / Duration)
    elif required_entity == "TIME_FREQ":
        if phrase_str:
            return (
                f"You mentioned {phrase_str}. How long or how frequently has this harassment/incident "
                f"been occurring (e.g., when did it start, or how often)?"
            )
        return (
            "How long or how frequently has this incident been occurring (e.g., when did it start, or how often)?"
        )

    # 4. LOCATION Dimension
    elif required_entity == "LOCATION":
        if phrase_str:
            return (
                f"Regarding {phrase_str}, where did this incident take place "
                f"(e.g., at home, workplace, educational institution, public space, or in transit)?"
            )
        return (
            "Where did this incident take place (e.g., at home, workplace, educational institution, public space, or in transit)?"
        )

    # 5. PLATFORM Dimension
    elif required_entity == "PLATFORM":
        if phrase_str:
            return (
                f"Regarding {phrase_str}, which platform or communication channel was used "
                f"(e.g., Instagram, WhatsApp, email, phone call, or website)?"
            )
        return (
            "Which platform or communication channel was used (e.g., Instagram, WhatsApp, email, phone call, or website)?"
        )

    # 6. INCIDENT_DETAILS Dimension
    elif required_entity == "INCIDENT_DETAILS":
        if phrase_str:
            return f"Could you share more specific details about what happened during {phrase_str}?"
        return "Could you provide more specific details about what happened during this incident?"

    # Fallback to provided DB fallback_text if unknown entity requirement
    if fallback_text and str(fallback_text).strip():
        return str(fallback_text).strip()

    return f"Could you provide additional details regarding your incident ({required_entity})?"
