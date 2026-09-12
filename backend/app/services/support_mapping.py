"""
Support Service Mapping Engine for Women's Safety Incident Analyzer.

Maps predicted incident categories and location (State / District) to verified support services
stored in the PostgreSQL support_services table.

Enforces strict anti-hallucination rules by returning ONLY empirical database records.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session as DBSession

from app.database import SessionLocal
from app.models.support_service import SupportService

logger = logging.getLogger("SupportMappingService")

# Exact mandatory fallback message required by technical specification
NO_MATCH_MESSAGE = "No matching support service was found in the available support-services database."

# Known valid input categories & aliases
VALID_CATEGORY_PATTERNS = {
    "DV": r"\b(DV|DOMESTIC VIOLENCE|DOWRY|SHELTER|OSC|ALL EMERGENCIES)\b",
    "DOMESTIC VIOLENCE": r"\b(DV|DOMESTIC VIOLENCE|DOWRY|SHELTER|OSC|ALL EMERGENCIES)\b",
    "SH": r"\b(SH|SEXUAL HARASSMENT|SEXUAL ASSAULT|RAPE|MODESTY|ALL EMERGENCIES)\b",
    "SEXUAL HARASSMENT": r"\b(SH|SEXUAL HARASSMENT|SEXUAL ASSAULT|RAPE|MODESTY|ALL EMERGENCIES)\b",
    "ST": r"\b(ST|STALKING|OBSCENE|CYBERCRIME|ALL EMERGENCIES)\b",
    "STALKING": r"\b(ST|STALKING|OBSCENE|CYBERCRIME|ALL EMERGENCIES)\b",
    "CA": r"\b(CA|CYBER|CYBERCRIME|OBSCENE|POSH|ALL EMERGENCIES)\b",
    "CYBER HARASSMENT": r"\b(CA|CYBER|CYBERCRIME|OBSCENE|POSH|ALL EMERGENCIES)\b",
    "CYBER ABUSE": r"\b(CA|CYBER|CYBERCRIME|OBSCENE|POSH|ALL EMERGENCIES)\b",
    "WH": r"\b(WH|POSH|WORKPLACE|SEXUAL HARASSMENT|ALL EMERGENCIES)\b",
    "WORKPLACE HARASSMENT": r"\b(WH|POSH|WORKPLACE|SEXUAL HARASSMENT|ALL EMERGENCIES)\b",
    "OV": r"\b(OV|OTHER|GENERAL|EMERGENCIES)\b",
    "OTHER HARASSMENT": r"\b(OV|OTHER|GENERAL|EMERGENCIES)\b",
    "OTHER SAFETY": r"\b(OV|OTHER|GENERAL|EMERGENCIES)\b",
    "GENERAL": r"\b(OV|OTHER|GENERAL|EMERGENCIES)\b",
    "FPM": r"\b(FPM|FALSE PROMISE|MARRIAGE|GENERAL|EMERGENCIES|ALL EMERGENCIES)\b",
    "FALSE PROMISE OF MARRIAGE": r"\b(FPM|FALSE PROMISE|MARRIAGE|GENERAL|EMERGENCIES|ALL EMERGENCIES)\b",
    "DOWRY": r"\b(DV|DOMESTIC VIOLENCE|DOWRY|SHELTER|OSC|ALL EMERGENCIES)\b",
    "DOWRY HARASSMENT": r"\b(DV|DOMESTIC VIOLENCE|DOWRY|SHELTER|OSC|ALL EMERGENCIES)\b",
    "EMERGENCY": r"\b(EMERGENCIES|POLICE|FIRE|MEDICAL)\b",
    "ALL EMERGENCIES": r"\b(EMERGENCIES|POLICE|FIRE|MEDICAL)\b",
}


def is_label_applicable(input_label: str, target_applicable_label: str) -> bool:
    """
    Determines if an input incident category matches the record's applicable_label field.
    Enforces anti-hallucination by rejecting unknown/unmapped input categories.
    """
    if not input_label or not target_applicable_label:
        return False

    cleaned_input = input_label.strip().upper()
    cleaned_target = target_applicable_label.strip().upper()

    pattern = VALID_CATEGORY_PATTERNS.get(cleaned_input)
    if not pattern:
        return False

    return bool(re.search(pattern, cleaned_target))


class SupportMappingService:
    """
    Service for querying PostgreSQL support_services table and matching verified support services.
    """

    def __init__(self, db: Optional[DBSession] = None):
        self.db = db

    def get_support_services(
        self,
        predicted_labels: Optional[List[str]] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        submission_id: Optional[str] = None,
        db_session: Optional[DBSession] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves verified support services matching incident categories and location.

        Args:
            predicted_labels: List of predicted BERT category strings/codes
            state: Optional state name string
            district: Optional district name string
            submission_id: Optional submission ID
            db_session: Optional database session

        Returns:
            Dict containing matched status and list of support_services dicts.
        """
        def fallback_response() -> Dict[str, Any]:
            return {
                "matched": False,
                "support_services": [],
                "message": NO_MATCH_MESSAGE,
            }

        # 1. Input Validation
        if not predicted_labels or not isinstance(predicted_labels, list):
            return fallback_response()

        cleaned_labels = [str(lbl).strip() for lbl in predicted_labels if lbl and str(lbl).strip()]
        if not cleaned_labels:
            return fallback_response()

        norm_state = str(state).strip().lower() if state and str(state).strip() else None
        norm_district = str(district).strip().lower() if district and str(district).strip() else None

        # 2. Database Session Management
        db = db_session or self.db
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # 3. Query All Support Services from PostgreSQL
            all_records = db.query(SupportService).all()
            if not all_records:
                return fallback_response()

            candidates = []

            # 4. Filter and Rank Records
            for rec in all_records:
                app_label = rec.applicable_label or ""
                
                # Check if record matches any predicted label
                matches_category = any(is_label_applicable(lbl, app_label) for lbl in cleaned_labels)
                if not matches_category:
                    continue

                rec_state = (rec.state or "").strip().lower()
                rec_district = (rec.district or "").strip().lower()

                location_score = 0
                is_national = rec_state in ["national", "all india"] or rec_district in ["all india", "national"]

                if norm_district and rec_district == norm_district:
                    location_score = 300  # Exact District match
                elif norm_state and rec_state == norm_state:
                    location_score = 200  # State match
                elif is_national:
                    location_score = 100  # National helpline fallback
                else:
                    location_score = 0    # Non-matching local service excluded

                if location_score > 0:
                    candidates.append((location_score, rec.id, rec))

            if not candidates:
                return fallback_response()

            # 5. Sort by location_score (descending) and ID (ascending)
            candidates.sort(key=lambda x: (-x[0], x[1]))

            # 6. Deduplicate & Format Output
            seen_keys = set()
            formatted_services = []

            for score, rec_id, rec in candidates:
                dedup_key = (rec.name, rec.service_type, rec.contact_number, rec.state, rec.district)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                formatted_services.append({
                    "service_name": rec.name,
                    "service_type": rec.service_type,
                    "contact_number": rec.contact_number,
                    "state": rec.state,
                    "district": rec.district,
                    "applicable_label": rec.applicable_label,
                    "source": "database",
                })

            if not formatted_services:
                return fallback_response()

            return {
                "matched": True,
                "support_services": formatted_services,
            }

        except Exception as e:
            logger.error(f"Error querying support services from PostgreSQL: {e}")
            return fallback_response()
        finally:
            if close_session and db:
                db.close()


def map_support_services(
    predicted_labels: Optional[List[str]] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    submission_id: Optional[str] = None,
    db: Optional[DBSession] = None,
) -> Dict[str, Any]:
    """
    Convenience function wrapper for SupportMappingService.
    """
    service = SupportMappingService(db=db)
    return service.get_support_services(
        predicted_labels=predicted_labels,
        state=state,
        district=district,
        submission_id=submission_id,
        db_session=db,
    )
