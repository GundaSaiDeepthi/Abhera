"""
Anti-Hallucination Safety & Provenance Validation Service.

Enforces strict verification of all legal and support service information
against empirical PostgreSQL database records before inclusion in reports or responses.

Core Principle:
"The model predicts and extracts; the database verifies and provides the final legal and support information."
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session as DBSession

from app.database import SessionLocal
from app.models.law import Law
from app.models.support_service import SupportService
from app.services.legal_mapping import LegalMappingService, NO_MATCH_MESSAGE as LEGAL_NO_MATCH_MESSAGE
from app.services.support_mapping import SupportMappingService, NO_MATCH_MESSAGE as SUPPORT_NO_MATCH_MESSAGE

logger = logging.getLogger("AntiHallucinationService")


class AntiHallucinationService:
    """
    Gating & Validation layer that verifies legal provisions and support services
    originated strictly from PostgreSQL laws and support_services tables.
    """

    def __init__(self, db: Optional[DBSession] = None):
        self.db = db

    def validate_legal_record(self, item: Dict[str, Any], db: DBSession) -> Tuple[bool, Optional[str]]:
        """
        Validates a single legal record against PostgreSQL laws table.

        Validation Criteria:
        1. Must contain required fields: act, section, description.
        2. Field values must NOT be empty or None.
        3. Must match an empirical row in PostgreSQL laws table verbatim.
        """
        if not isinstance(item, dict):
            return False, "Invalid record format: not a dictionary"

        act = item.get("act")
        section = item.get("section")
        description = item.get("description")

        if not act or not str(act).strip():
            return False, "Missing or empty 'act' field"
        if not section or not str(section).strip():
            return False, "Missing or empty 'section' field"
        if not description or not str(description).strip():
            return False, "Missing or empty 'description' field"

        # Query PostgreSQL laws table for exact matching record
        db_match = (
            db.query(Law)
            .filter(
                Law.act_name == str(act).strip(),
                Law.section_number == str(section).strip(),
            )
            .first()
        )

        if not db_match:
            return False, f"Fabricated legal record rejected: Act '{act}' Section '{section}' not found in PostgreSQL laws table"

        # Verify description matches database verbatim
        if db_match.section_text.strip() != str(description).strip():
            return False, f"Modified legal description rejected for Act '{act}' Section '{section}'"

        return True, None

    def validate_support_record(self, item: Dict[str, Any], db: DBSession) -> Tuple[bool, Optional[str]]:
        """
        Validates a single support service record against PostgreSQL support_services table.

        Validation Criteria:
        1. Must contain required fields: service_name, contact_number, state, district.
        2. Field values must NOT be empty or None.
        3. Must match an empirical row in PostgreSQL support_services table verbatim.
        """
        if not isinstance(item, dict):
            return False, "Invalid record format: not a dictionary"

        name = item.get("service_name")
        contact = item.get("contact_number")
        state = item.get("state")
        district = item.get("district")

        if not name or not str(name).strip():
            return False, "Missing or empty 'service_name' field"
        if not contact or not str(contact).strip():
            return False, "Missing or empty 'contact_number' field"
        if not state or not str(state).strip():
            return False, "Missing or empty 'state' field"
        if not district or not str(district).strip():
            return False, "Missing or empty 'district' field"

        # Query PostgreSQL support_services table for exact matching record
        db_match = (
            db.query(SupportService)
            .filter(
                SupportService.name == str(name).strip(),
                SupportService.contact_number == str(contact).strip(),
            )
            .first()
        )

        if not db_match:
            return False, f"Fabricated support service rejected: '{name}' ({contact}) not found in PostgreSQL support_services table"

        return True, None

    def validate_and_gate_results(
        self,
        predicted_labels: Optional[List[str]] = None,
        legal_input: Optional[Dict[str, Any]] = None,
        support_input: Optional[Dict[str, Any]] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        submission_id: Optional[str] = None,
        db_session: Optional[DBSession] = None,
    ) -> Dict[str, Any]:
        """
        Main validation pipeline.
        Validates legal provisions and support services, gating out fabricated entries
        and attaching internal audit metrics.
        """
        db = db_session or self.db
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # 1. Obtain Legal Provisions if not passed
            if legal_input is None and predicted_labels:
                legal_service = LegalMappingService(db=db)
                legal_input = legal_service.get_legal_provisions(predicted_labels=predicted_labels, db_session=db)
            elif legal_input is None:
                legal_input = {"matched": False, "legal_information": [], "message": LEGAL_NO_MATCH_MESSAGE}

            # 2. Obtain Support Services if not passed
            if support_input is None and predicted_labels:
                support_service = SupportMappingService(db=db)
                support_input = support_service.get_support_services(
                    predicted_labels=predicted_labels, state=state, district=district, db_session=db
                )
            elif support_input is None:
                support_input = {"matched": False, "support_services": [], "message": SUPPORT_NO_MATCH_MESSAGE}

            # 3. Validate Legal Information
            raw_legal_list = legal_input.get("legal_information", []) if isinstance(legal_input, dict) else []
            accepted_legal = []
            legal_audit = {"received": len(raw_legal_list), "accepted": 0, "rejected": 0, "rejection_reasons": []}

            for item in raw_legal_list:
                valid, reason = self.validate_legal_record(item, db)
                if valid:
                    accepted_legal.append(item)
                    legal_audit["accepted"] += 1
                else:
                    legal_audit["rejected"] += 1
                    if reason:
                        legal_audit["rejection_reasons"].append(reason)

            # 4. Validate Support Services
            raw_support_list = support_input.get("support_services", []) if isinstance(support_input, dict) else []
            accepted_support = []
            support_audit = {"received": len(raw_support_list), "accepted": 0, "rejected": 0, "rejection_reasons": []}

            for item in raw_support_list:
                valid, reason = self.validate_support_record(item, db)
                if valid:
                    accepted_support.append(item)
                    support_audit["accepted"] += 1
                else:
                    support_audit["rejected"] += 1
                    if reason:
                        support_audit["rejection_reasons"].append(reason)

            # 5. Build Gated Output Payload
            legal_matched = len(accepted_legal) > 0
            support_matched = len(accepted_support) > 0

            return {
                "legal_matched": legal_matched,
                "legal_information": accepted_legal,
                "legal_message": None if legal_matched else LEGAL_NO_MATCH_MESSAGE,
                "support_matched": support_matched,
                "support_services": accepted_support,
                "support_message": None if support_matched else SUPPORT_NO_MATCH_MESSAGE,
                "audit": {
                    "legal": legal_audit,
                    "support": support_audit,
                },
            }

        finally:
            if close_session and db:
                db.close()


def validate_payload(
    predicted_labels: Optional[List[str]] = None,
    legal_input: Optional[Dict[str, Any]] = None,
    support_input: Optional[Dict[str, Any]] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    submission_id: Optional[str] = None,
    db: Optional[DBSession] = None,
) -> Dict[str, Any]:
    """
    Convenience function wrapper for AntiHallucinationService.
    """
    service = AntiHallucinationService(db=db)
    return service.validate_and_gate_results(
        predicted_labels=predicted_labels,
        legal_input=legal_input,
        support_input=support_input,
        state=state,
        district=district,
        submission_id=submission_id,
        db_session=db,
    )
