# Incident Report Generator Technical Specification & Architecture

## 1. Purpose & Core Principles

The **Report Generator Service** compiles a structured, transparent, non-generative final incident report by gathering data from all completed pipeline components: BERT Classification, NER Entity Extraction, Dynamic Question Engine, and Anti-Hallucination validated Legal & Support Mapping.

> **Fundamental Architectural Principle**:  
> *"The model predicts and extracts; the database verifies and provides the final legal and support information."*

The Report Generator **never invents** legal provisions, support helplines, incident categories, entities, or factual statements. It strictly assembles data produced and validated upstream.

---

## 2. End-to-End System Data Flow

```
                      User Conversation & Narrative
                                   │
                                   ▼
                    BERT Multi-Label Classification
                        (e.g., ["DV", "SH"])
                                   │
                                   ▼
                    NER Entity Extraction Spans
                    (PERP_REL, LOCATION, etc.)
                                   │
                                   ▼
                        Dynamic Question Engine
                   (Collected User Answers & Details)
                                   │
                                   ▼
                Legal & Support Service Mapping Engine
               (PostgreSQL Database Lookup on `laws` & `support_services`)
                                   │
                                   ▼
                   Anti-Hallucination Validation Layer
                (Gating, Verification, Audit Logging)
                                   │
                                   ▼
                       REPORT GENERATOR SERVICE
              (Assembles 10 Required Conceptual Sections)
                                   │
                                   ▼
                        Final Incident Report JSON
```

---

## 3. Ten Required Conceptual Sections

The generated report contains 10 required sections in exact conceptual order:

| Section # | Section Key | Description | Source / Provenance |
| :---: | :--- | :--- | :--- |
| **1** | `incident_summary` | Concise factual summary of user narrative | User Narrative Text |
| **2** | `identified_incident_type` | Predicted BERT incident categories | BERT Classifier Output (`["DV", "SH"]`) |
| **3** | `extracted_information` | Extracted entity spans by category | BERT NER Model (`PERP_REL`, `LOCATION`, etc.) |
| **4** | `user_provided_details` | User answers collected dynamically | Dynamic Question Engine (`submission_question_answers`) |
| **5** | `relevant_legal_information` | Validated statutory provisions | PostgreSQL `laws` (via `AntiHallucinationService`) |
| **6** | `suggested_next_steps` | Safe, non-generative general guidance | Static Pre-Approved Safety Guidance List |
| **7** | `available_support_services` | Verified emergency helplines & shelters | PostgreSQL `support_services` (via `AntiHallucinationService`) |
| **8** | `evidence_information_notes` | Evidence notes and extracted evidence spans | User Submission / Extracted Evidence Spans |
| **9** | `model_prediction_information` | Model metadata & provenance declaration | BERT & NER Model Metadata |
| **10** | `disclaimer` | Legal non-advice & informational disclaimer | Standard Pre-Approved Legal Disclaimer |

---

## 4. Fallback Behavior & Exact Fallback Strings

When data is missing or no matching database rows exist, the Report Generator enforces exact fallback strings:

- **Missing Legal Match**: Returns **EXACTLY**:
  `"Information not available in the provided knowledge base."`
- **Missing Support Match**: Returns **EXACTLY**:
  `"No matching support service was found in the available support-services database."`
- **Missing Incident Narrative**: Returns **EXACTLY**:
  `"Incident summary is not available from the provided information."`
- **Missing BERT Categories**: Returns:
  `{"labels": [], "message": "Incident type could not be determined from the available model output."}`
- **Missing Entity Spans**: Returns empty list `[]` for that entity slot (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`, `EVIDENCE`, `LAW_SEC`).

---

## 5. Security & Anti-Hallucination Compliance

1. **Anti-Hallucination Integration**: The Report Generator passes all legal and support results through `AntiHallucinationService.validate_and_gate_results()`. Any item not found in PostgreSQL `laws` or `support_services` is rejected before report assembly.
2. **Zero Generative Code**: Contains zero LLM calls, zero generative text scripts, and zero hardcoded legal maps (no `if category == "DV": return "Section XYZ"`).
3. **Value Preservation**: Database column values (`act_name`, `section_number`, `section_text`, `name`, `contact_number`, `state`, `district`) are preserved verbatim.
4. **Offline Operation**: Runs 100% offline without external web requests, external APIs, or web scraping.

---

## 6. Test Suite Coverage (20 Tests)

All 20 unit and integration test cases in [`backend/tests/test_report_generator.py`](file:///D:/Abhera(Mini)/backend/tests/test_report_generator.py) passed:

1. `test_01_valid_complete_report_generation`: Assembles full 10-section report.
2. `test_02_incident_summary_uses_user_info_only`: Uses user text only; falls back when empty.
3. `test_03_bert_labels_preserved`: Preserves BERT category labels.
4. `test_04_multi_label_incidents_preserved`: Preserves multi-label incident outputs.
5. `test_05_ner_entities_preserved`: Preserves extracted NER entity spans in correct slots.
6. `test_06_user_answers_included`: Includes collected questionnaire answers.
7. `test_07_valid_legal_database_records_appear`: Includes validated PostgreSQL legal records.
8. `test_08_fabricated_legal_records_cannot_appear`: Injected fake laws are rejected and excluded.
9. `test_09_exact_legal_fallback_appears`: Returns exact legal fallback string when no laws match.
10. `test_10_valid_support_services_appear`: Includes validated PostgreSQL support service records.
11. `test_11_fabricated_support_services_cannot_appear`: Injected fake helplines are rejected and excluded.
12. `test_12_exact_support_fallback_appears`: Returns exact support fallback string when no helplines match.
13. `test_13_unknown_category_does_not_create_legal`: Returns exact legal fallback string for unknown categories.
14. `test_14_unknown_category_does_not_create_support`: Returns exact support fallback string for unknown categories.
15. `test_15_missing_entity_categories_produce_empty_lists`: Empty entity slots return `[]`.
16. `test_16_missing_user_answers_not_converted_to_facts`: Unanswered questions excluded.
17. `test_17_model_prediction_info_separated_from_db_info`: Transparently separates AI predictions from DB knowledge.
18. `test_18_no_external_api_dependency`: Operates 100% offline.
19. `test_19_disclaimer_is_present`: Includes standard legal disclaimer.
20. `test_20_report_contains_all_ten_sections`: Verifies presence of all 10 conceptual sections.

---

## 7. Sample Report Output JSON

```json
{
    "submission_id": "SUB-REPORT-001",
    "incident_summary": "My former colleague harassed me at the workplace.",
    "identified_incident_type": {
        "labels": ["Workplace Harassment"],
        "message": null
    },
    "extracted_information": {
        "PERP_REL": ["former colleague"],
        "LOCATION": ["workplace"],
        "TIME_FREQ": [],
        "PLATFORM": [],
        "EVIDENCE": [],
        "LAW_SEC": []
    },
    "user_provided_details": [
        {
            "question_id": "Q_SAFETY_01",
            "question": "Are you safe?",
            "answer": "Yes, I am safe."
        }
    ],
    "relevant_legal_information": [
        {
            "act": "POSH_ACT",
            "section": "2n",
            "description": "Sexual harassment includes any one or more of the following unwelcome acts or behaviour...",
            "applicable_label": "WH",
            "source": "database"
        }
    ],
    "suggested_next_steps": [
        "Preserve all relevant evidence, messages, call logs, screenshots, and documents.",
        "Keep copies of any formal communications or written notices.",
        "Consider contacting an appropriate support service or helpline from the verified database.",
        "If you are in immediate physical danger, seek emergency assistance through local police (112) or emergency services."
    ],
    "available_support_services": [
        {
            "service_name": "SHe-Box (Sexual Harassment electronic Box)",
            "service_type": "Portal",
            "contact_number": "https://shebox.wcd.gov.in",
            "state": "National",
            "district": "All India",
            "applicable_label": "WH(POSH act),CA",
            "source": "database"
        }
    ],
    "evidence_information_notes": [],
    "model_prediction_information": {
        "bert_classifier": {
            "model_name": "bert-base-uncased",
            "task": "Multi-Label Incident Classification",
            "predicted_labels": ["Workplace Harassment"]
        },
        "ner_extractor": {
            "model_name": "bert-base-uncased-ner",
            "task": "Token Classification Entity Extraction",
            "extracted_entity_types": ["PERP_REL", "LOCATION"]
        },
        "data_provenance": "Legal provisions and support services are retrieved directly from verified PostgreSQL database records, NOT generated by AI models."
    },
    "disclaimer": "This system provides informational assistance based on the available dataset, trained models, and verified database records. It does not replace professional legal advice, emergency services, law enforcement, or other qualified support."
}
```
