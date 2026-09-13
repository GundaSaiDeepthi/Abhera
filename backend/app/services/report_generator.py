"""
Incident Report Generator Service.

Assembles a structured, transparent, non-generative incident report from
BERT classifications, NER extracted entities, user questionnaire answers,
and Anti-Hallucination validated PostgreSQL legal and support database records.

Core Principle:
"The model predicts and extracts; the database verifies and provides the final legal and support information."
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session as DBSession

from app.database import SessionLocal
from app.models.incident_submission import IncidentSubmission
from app.models.question import Question, SubmissionQuestionAnswer
from app.models.report import Report
from app.models.session import Session as UserSession
from app.services.anti_hallucination import (
    validate_payload,
    LEGAL_NO_MATCH_MESSAGE,
    SUPPORT_NO_MATCH_MESSAGE,
)

logger = logging.getLogger("ReportGeneratorService")

DEFAULT_DISCLAIMER = (
    "This system provides informational assistance based on the available dataset, trained models, "
    "and verified database records. It does not replace professional legal advice, emergency services, "
    "law enforcement, or other qualified support."
)

DEFAULT_NEXT_STEPS = [
    "Preserve all relevant evidence, messages, call logs, screenshots, and documents.",
    "Keep copies of any formal communications or written notices.",
    "Consider contacting an appropriate support service or helpline from the verified database.",
    "If you are in immediate physical danger, seek emergency assistance through local police (112) or emergency services.",
]


class ReportGeneratorService:
    """
    Generates structured, non-generative final incident reports.
    """

    def __init__(self, db: Optional[DBSession] = None):
        self.db = db

    def generate_report(
        self,
        submission_id: Optional[str] = None,
        narrative_text: Optional[str] = None,
        bert_results: Optional[Any] = None,
        ner_entities: Optional[List[Dict[str, Any]]] = None,
        user_answers: Optional[Any] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        evidence_notes: Optional[List[str]] = None,
        db_session: Optional[DBSession] = None,
    ) -> Dict[str, Any]:
        """
        Assembles all 10 required conceptual report sections into a structured report dictionary.
        """
        db = db_session or self.db
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # Load from DB if submission_id provided and missing explicit fields
            if submission_id and db is not None:
                submission_rec = (
                    db.query(IncidentSubmission)
                    .filter(IncidentSubmission.submission_id == submission_id)
                    .first()
                )
                if submission_rec:
                    if narrative_text is None:
                        narrative_text = submission_rec.narrative_text
                    if bert_results is None and submission_rec.predicted_labels:
                        bert_results = submission_rec.predicted_labels
                    if ner_entities is None and submission_rec.extracted_entities:
                        ner_entities = submission_rec.extracted_entities
                    if state is None:
                        state = submission_rec.state
                    if district is None:
                        district = submission_rec.district

                # Fallback check on UserSession table if submission fields are missing
                sess_rec = None
                if submission_rec and submission_rec.session_id:
                    sess_rec = db.query(UserSession).filter(UserSession.session_id == submission_rec.session_id).first()
                elif submission_id:
                    sess_rec = db.query(UserSession).filter(UserSession.submission_id == submission_id).first()

                if sess_rec:
                    if not bert_results and sess_rec.predicted_labels:
                        bert_results = sess_rec.predicted_labels
                    if not ner_entities and sess_rec.entities:
                        ner_entities = sess_rec.entities
                    if user_answers is None and sess_rec.answers:
                        user_answers = sess_rec.answers

            # ------------------------------------------------------------------
            # SECTION 1 — INCIDENT SUMMARY
            # ------------------------------------------------------------------
            if narrative_text and str(narrative_text).strip():
                incident_summary = str(narrative_text).strip()
            else:
                incident_summary = "Incident summary is not available from the provided information."

            # ------------------------------------------------------------------
            # SECTION 2 — IDENTIFIED INCIDENT TYPE
            # ------------------------------------------------------------------
            predicted_labels_list = []
            if bert_results:
                if isinstance(bert_results, dict):
                    raw_labels = bert_results.get("labels") or bert_results.get("predicted_labels") or []
                    if isinstance(raw_labels, list):
                        for l in raw_labels:
                            if isinstance(l, dict):
                                val = l.get("name") or l.get("code") or l.get("label")
                            else:
                                val = str(l)
                            if val and str(val).strip() and str(val).strip() not in predicted_labels_list:
                                predicted_labels_list.append(str(val).strip())
                elif isinstance(bert_results, list):
                    for l in bert_results:
                        if isinstance(l, dict):
                            val = l.get("name") or l.get("code") or l.get("label")
                        else:
                            val = str(l)
                        if val and str(val).strip() and str(val).strip() not in predicted_labels_list:
                            predicted_labels_list.append(str(val).strip())

            if predicted_labels_list:
                identified_incident_type = {
                    "labels": predicted_labels_list,
                    "message": None,
                }
            else:
                identified_incident_type = {
                    "labels": [],
                    "message": "Incident type could not be determined from the available model output.",
                }

            # ------------------------------------------------------------------
            # SECTION 3 — EXTRACTED INFORMATION (NER Entities)
            # ------------------------------------------------------------------
            extracted_info = {
                "PERP_REL": [],
                "LOCATION": [],
                "TIME_FREQ": [],
                "PLATFORM": [],
                "EVIDENCE": [],
                "LAW_SEC": [],
            }

            if ner_entities:
                if isinstance(ner_entities, list):
                    for ent in ner_entities:
                        if isinstance(ent, dict):
                            lbl = ent.get("label") or ent.get("entity_type")
                            txt = ent.get("text") or ent.get("entity_text")
                            if lbl in extracted_info and txt and str(txt).strip():
                                val = str(txt).strip()
                                if val not in extracted_info[lbl]:
                                    extracted_info[lbl].append(val)
                elif isinstance(ner_entities, dict):
                    for lbl, items in ner_entities.items():
                        if lbl in extracted_info:
                            if isinstance(items, list):
                                for item in items:
                                    if isinstance(item, dict):
                                        txt = item.get("text") or item.get("entity_text")
                                    else:
                                        txt = str(item)
                                    if txt and str(txt).strip():
                                        val = str(txt).strip()
                                        if val not in extracted_info[lbl]:
                                            extracted_info[lbl].append(val)
                            elif isinstance(items, str) and items.strip():
                                val = items.strip()
                                if val not in extracted_info[lbl]:
                                    extracted_info[lbl].append(val)

            # ------------------------------------------------------------------
            # SECTION 4 — USER-PROVIDED DETAILS (Question Answers)
            # ------------------------------------------------------------------
            formatted_user_details = []
            if user_answers:
                if isinstance(user_answers, list):
                    for item in user_answers:
                        if isinstance(item, dict) and item.get("answer"):
                            q_id = item.get("question_id", "Q_CUSTOM")
                            q_text = item.get("question") or item.get("question_text")
                            if (not q_text or q_text == "Question" or str(q_text).startswith("Question ")) and db and q_id and q_id != "Q_CUSTOM":
                                q_rec = db.query(Question).filter(Question.question_id == str(q_id)).first()
                                if q_rec and q_rec.question_text:
                                    q_text = q_rec.question_text
                            if not q_text:
                                q_text = f"Question {q_id}"
                            formatted_user_details.append({
                                "question_id": str(q_id),
                                "question": str(q_text).strip(),
                                "answer": str(item.get("answer")).strip(),
                            })
                elif isinstance(user_answers, dict):
                    for q_id, ans in user_answers.items():
                        if ans and str(ans).strip():
                            q_text = None
                            if db and q_id:
                                q_rec = db.query(Question).filter(Question.question_id == str(q_id)).first()
                                if q_rec and q_rec.question_text:
                                    q_text = q_rec.question_text
                            if not q_text:
                                q_text = f"Question {q_id}"
                            formatted_user_details.append({
                                "question_id": str(q_id),
                                "question": str(q_text).strip(),
                                "answer": str(ans).strip(),
                            })

            if not formatted_user_details and submission_id and db is not None:
                sqa_recs = (
                    db.query(SubmissionQuestionAnswer)
                    .filter(SubmissionQuestionAnswer.submission_id == submission_id)
                    .order_by(SubmissionQuestionAnswer.id.asc())
                    .all()
                )
                if sqa_recs:
                    for sqa in sqa_recs:
                        if sqa.answer and str(sqa.answer).strip():
                            q_text = sqa.question.question_text if (sqa.question and sqa.question.question_text) else f"Question {sqa.question_id}"
                            formatted_user_details.append({
                                "question_id": str(sqa.question_id),
                                "question": str(q_text).strip(),
                                "answer": str(sqa.answer).strip(),
                            })

            # ------------------------------------------------------------------
            # SECTION 5 & 7 — LEGAL & SUPPORT MAPPING via Anti-Hallucination Layer
            # ------------------------------------------------------------------
            validated_payload = validate_payload(
                predicted_labels=predicted_labels_list,
                state=state,
                district=district,
                db=db,
            )

            # Relevant Legal Information
            if validated_payload["legal_matched"] and validated_payload["legal_information"]:
                relevant_legal_information = validated_payload["legal_information"]
            else:
                relevant_legal_information = LEGAL_NO_MATCH_MESSAGE

            # Available Support Services
            if validated_payload["support_matched"] and validated_payload["support_services"]:
                available_support_services = validated_payload["support_services"]
            else:
                available_support_services = SUPPORT_NO_MATCH_MESSAGE

            # ------------------------------------------------------------------
            # SECTION 6 — SUGGESTED NEXT STEPS
            # ------------------------------------------------------------------
            suggested_next_steps = list(DEFAULT_NEXT_STEPS)

            # ------------------------------------------------------------------
            # SECTION 8 — EVIDENCE / INFORMATION NOTES
            # ------------------------------------------------------------------
            formatted_evidence_notes = []
            if evidence_notes and isinstance(evidence_notes, list):
                formatted_evidence_notes.extend([str(n).strip() for n in evidence_notes if str(n).strip()])

            # Include any extracted EVIDENCE NER entity spans if present
            if extracted_info["EVIDENCE"]:
                for ev in extracted_info["EVIDENCE"]:
                    if ev not in formatted_evidence_notes:
                        formatted_evidence_notes.append(f"Extracted evidence mention: {ev}")

            # ------------------------------------------------------------------
            # SECTION 9 — MODEL PREDICTION INFORMATION
            # ------------------------------------------------------------------
            model_prediction_info = {
                "bert_classifier": {
                    "model_name": "bert-base-uncased",
                    "task": "Multi-Label Incident Classification",
                    "predicted_labels": predicted_labels_list,
                },
                "ner_extractor": {
                    "model_name": "bert-base-uncased-ner",
                    "task": "Token Classification Entity Extraction",
                    "extracted_entity_types": [k for k, v in extracted_info.items() if len(v) > 0],
                },
                "data_provenance": "Legal provisions and support services are retrieved directly from verified PostgreSQL database records, NOT generated by AI models.",
            }

            # ------------------------------------------------------------------
            # SECTION 10 — DISCLAIMER
            # ------------------------------------------------------------------
            disclaimer = DEFAULT_DISCLAIMER

            # ------------------------------------------------------------------
            # ASSEMBLE COMPLETE REPORT DICTIONARY
            # ------------------------------------------------------------------
            report_dict = {
                "submission_id": submission_id,
                "incident_summary": incident_summary,
                "identified_incident_type": identified_incident_type,
                "extracted_information": extracted_info,
                "user_provided_details": formatted_user_details,
                "relevant_legal_information": relevant_legal_information,
                "suggested_next_steps": suggested_next_steps,
                "available_support_services": available_support_services,
                "evidence_information_notes": formatted_evidence_notes,
                "model_prediction_information": model_prediction_info,
                "disclaimer": disclaimer,
                "audit": validated_payload.get("audit", {}),
            }

            # Save to PostgreSQL reports table if submission_id exists in incident_submissions
            if submission_id and db is not None and submission_rec:
                try:
                    existing_report = (
                        db.query(Report)
                        .filter(Report.submission_id == submission_id)
                        .first()
                    )
                    if existing_report:
                        existing_report.report_data = report_dict
                    else:
                        new_report = Report(submission_id=submission_id, report_data=report_dict)
                        db.add(new_report)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.warning(f"Could not persist report to PostgreSQL: {e}")

            return report_dict

        finally:
            if close_session and db:
                db.close()


def generate_incident_report(
    submission_id: Optional[str] = None,
    narrative_text: Optional[str] = None,
    bert_results: Optional[Any] = None,
    ner_entities: Optional[List[Dict[str, Any]]] = None,
    user_answers: Optional[Any] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    evidence_notes: Optional[List[str]] = None,
    db: Optional[DBSession] = None,
) -> Dict[str, Any]:
    """
    Convenience function wrapper for ReportGeneratorService.
    """
    service = ReportGeneratorService(db=db)
    return service.generate_report(
        submission_id=submission_id,
        narrative_text=narrative_text,
        bert_results=bert_results,
        ner_entities=ner_entities,
        user_answers=user_answers,
        state=state,
        district=district,
        evidence_notes=evidence_notes,
        db_session=db,
    )
