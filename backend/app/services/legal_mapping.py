"""
Legal Mapping Service for Women's Safety Incident Analyzer.

Maps BERT predicted incident categories to official legal provisions from the
PostgreSQL laws table. Enforces strict anti-hallucination rules by returning ONLY
empirically stored database records.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session as DBSession

from app.database import SessionLocal
from app.models.law import Law

logger = logging.getLogger("LegalMappingService")

# Exact mandatory fallback message
NO_MATCH_MESSAGE = "Information not available in the provided knowledge base."

# Label Normalization Map: Maps human-readable labels or shortcodes to laws.applicable_label values
LABEL_TO_APPLICABLE_CODE = {
    "DV": "DV",
    "DOMESTIC VIOLENCE": "DV",
    "SH": "SH",
    "SEXUAL HARASSMENT": "SH",
    "ST": "ST",
    "STALKING": "ST",
    "CA": "CA",
    "CYBER HARASSMENT": "CA",
    "CYBER ABUSE": "CA",
    "WH": "WH",
    "WORKPLACE HARASSMENT": "WH",
    "OV": "OV",
    "OTHER HARASSMENT": "OV",
    "OTHER / GENERAL HARASSMENT": "OV",
    "OTHER SAFETY": "OV",
    "GENERAL": "OV",
    "FPM": "FPM",
    "FALSE PROMISE OF MARRIAGE": "FPM",
    "DOWRY": "DOWRY",
    "DOWRY HARASSMENT": "DOWRY",
    "DOWRY PROHIBITION ACT": "DOWRY",
}


def normalize_to_applicable_label(label: str) -> Optional[str]:
    """
    Normalizes input label string to the exact applicable_label stored in the laws table.
    Returns None if label is empty, invalid, or unmapped.
    """
    if not label or not isinstance(label, str):
        return None
    cleaned = label.strip().upper()
    return LABEL_TO_APPLICABLE_CODE.get(cleaned)


class LegalMappingService:
    """
    Service for querying PostgreSQL laws table and mapping incident labels to legal provisions.
    """

    def __init__(self, db: Optional[DBSession] = None):
        self.db = db

    def get_legal_provisions(
        self,
        predicted_labels: Optional[List[str]] = None,
        incident_information: Optional[Dict[str, Any]] = None,
        submission_id: Optional[str] = None,
        db_session: Optional[DBSession] = None,
    ) -> Dict[str, Any]:
        """
        Maps predicted incident labels to legal provisions stored in the PostgreSQL database.

        Args:
            predicted_labels: List of predicted BERT category strings/codes
            incident_information: Optional dict of collected incident details
            submission_id: Optional submission ID
            db_session: Optional database session

        Returns:
            Dict containing matched status and list of legal_information dicts.
        """
        def fallback_response() -> Dict[str, Any]:
            return {
                "matched": False,
                "legal_information": [],
                "message": NO_MATCH_MESSAGE,
            }

        # 1. Input Validation
        if not predicted_labels or not isinstance(predicted_labels, list):
            return fallback_response()

        # 2. Normalize and Deduplicate Input Labels
        normalized_codes = []
        for lbl in predicted_labels:
            code = normalize_to_applicable_label(lbl)
            if code and code not in normalized_codes:
                normalized_codes.append(code)

        if not normalized_codes:
            return fallback_response()

        # 3. Database Session Management
        db = db_session or self.db
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # 4. Query PostgreSQL Laws Table (Parameterized ORM Query)
            raw_records = (
                db.query(Law)
                .filter(Law.applicable_label.in_(normalized_codes))
                .order_by(Law.id.asc())
                .all()
            )

            if not raw_records:
                return fallback_response()

            # 5. Format Output & Deduplicate (Preserve exact DB field values)
            seen_keys = set()
            legal_info_list = []

            for rec in raw_records:
                dedup_key = (rec.act_name, rec.section_number, rec.applicable_label)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                legal_info_list.append({
                    "act": rec.act_name,
                    "section": str(rec.section_number),
                    "description": rec.section_text,
                    "applicable_label": rec.applicable_label,
                    "source": "database",
                })

            if not legal_info_list:
                return fallback_response()

            return {
                "matched": True,
                "legal_information": legal_info_list,
            }

        except Exception as e:
            logger.error(f"Error querying legal provisions from PostgreSQL: {e}")
            return fallback_response()
        finally:
            if close_session and db:
                db.close()


def map_legal_provisions(
    predicted_labels: Optional[List[str]] = None,
    incident_information: Optional[Dict[str, Any]] = None,
    submission_id: Optional[str] = None,
    db: Optional[DBSession] = None,
) -> Dict[str, Any]:
    """
    Convenience function wrapper for LegalMappingService.
    """
    service = LegalMappingService(db=db)
    return service.get_legal_provisions(
        predicted_labels=predicted_labels,
        incident_information=incident_information,
        submission_id=submission_id,
        db_session=db,
    )
