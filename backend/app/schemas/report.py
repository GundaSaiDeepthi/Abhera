"""
Pydantic Schemas for Report API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReportGenerateRequest(BaseModel):
    submission_id: str = Field(..., description="Unique incident submission ID (e.g. SUB-12345678)")
    state: Optional[str] = Field(None, description="Optional jurisdiction state name")
    district: Optional[str] = Field(None, description="Optional jurisdiction district name")
    narrative_text: Optional[str] = Field(None, description="Optional override narrative text")
    bert_results: Optional[Any] = Field(None, description="Optional override predicted labels")
    ner_entities: Optional[List[Dict[str, Any]]] = Field(None, description="Optional override extracted entities")
    user_answers: Optional[Any] = Field(None, description="Optional override user answers")


class ReportResponse(BaseModel):
    submission_id: Optional[str] = None
    incident_summary: str
    identified_incident_type: Dict[str, Any]
    extracted_information: Dict[str, List[str]]
    user_provided_details: List[Dict[str, Any]]
    relevant_legal_information: Any
    suggested_next_steps: List[str]
    available_support_services: Any
    evidence_information_notes: List[str]
    model_prediction_information: Dict[str, Any]
    disclaimer: str
    audit: Optional[Dict[str, Any]] = None
