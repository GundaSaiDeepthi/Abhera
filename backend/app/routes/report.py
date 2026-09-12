"""
Report API Router for Women's Safety Legal Assistance System.

Exposes RESTful endpoints for generating and retrieving structured incident reports.

Endpoints:
- POST /api/report/generate
- GET  /api/report/{submission_id}
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.incident_submission import IncidentSubmission
from app.models.report import Report
from app.schemas.report import ReportGenerateRequest, ReportResponse
from app.services.report_generator import generate_incident_report

logger = logging.getLogger("ReportAPI")

router = APIRouter(prefix="/api/report", tags=["Report"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Incident Report",
    description="Generates and persists a structured final incident report for a given submission ID.",
)
def generate_report_endpoint(
    payload: ReportGenerateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    POST /api/report/generate

    Validates submission_id and generates a structured report using ReportGeneratorService.
    Relying on Anti-Hallucination layer for PostgreSQL legal & support provenance.
    """
    submission_id = payload.submission_id.strip() if payload.submission_id else None
    if not submission_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'submission_id' is required.",
        )

    # Validate submission existence in PostgreSQL incident_submissions if no explicit narrative override provided
    submission_rec = (
        db.query(IncidentSubmission)
        .filter(IncidentSubmission.submission_id == submission_id)
        .first()
    )

    if not submission_rec and not payload.narrative_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident submission '{submission_id}' not found.",
        )

    try:
        report = generate_incident_report(
            submission_id=submission_id,
            narrative_text=payload.narrative_text,
            bert_results=payload.bert_results,
            ner_entities=payload.ner_entities,
            user_answers=payload.user_answers,
            state=payload.state,
            district=payload.district,
            db=db,
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for submission '{submission_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the incident report.",
        )


@router.get(
    "/{submission_id}",
    response_model=ReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Generated Incident Report",
    description="Retrieves the generated incident report for a submission ID.",
)
def get_report_endpoint(
    submission_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    GET /api/report/{submission_id}

    Retrieves a stored report from PostgreSQL reports table or generates it if submission exists.
    """
    clean_sub_id = submission_id.strip() if submission_id else None
    if not clean_sub_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid submission ID.",
        )

    # 1. Check PostgreSQL reports table
    report_rec = (
        db.query(Report)
        .filter(Report.submission_id == clean_sub_id)
        .first()
    )

    if report_rec and report_rec.report_data:
        return report_rec.report_data

    # 2. Check if submission exists to generate report dynamically
    submission_rec = (
        db.query(IncidentSubmission)
        .filter(IncidentSubmission.submission_id == clean_sub_id)
        .first()
    )

    if not submission_rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report or submission '{clean_sub_id}' not found.",
        )

    # Generate report dynamically for existing submission
    try:
        report = generate_incident_report(submission_id=clean_sub_id, db=db)
        return report
    except Exception as e:
        logger.error(f"Error retrieving/generating report for '{clean_sub_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while retrieving the report.",
        )
