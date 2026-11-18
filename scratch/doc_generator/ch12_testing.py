"""
Module for Chapter 12: Testing and Evaluation
"""

def get_chapter_12():
    return r'''# CHAPTER 12: Testing and Evaluation

---

## 12.1 Testing Philosophy and Verification Standards

In safety-critical legal informatics, automated software testing is not an optional afterthought—it is a vital safeguard against regressions, incorrect legal citations, state machine deadlock, and memory leaks.

ABHERA implements a comprehensive, multi-layer testing strategy:
1. **Unit Testing**: Testing individual functions, tokenizers, BIO parsers, and slot rules in complete isolation.
2. **Integration Testing**: Testing communication between API routers, Pydantic schemas, and SQLAlchemy database tables using FastAPI's `TestClient`.
3. **Machine Learning Model Auditing**: Evaluating held-out test records against ground-truth labels using exact micro/macro precision, recall, and F1 calculations.
4. **Anti-Hallucination Stress Testing**: Deliberately injecting malicious, invalid, or hallucinated legal sections to confirm that the gating layer intercepts and rejects 100% of unverified claims.
5. **End-to-End System Simulation**: Simulating multi-turn conversational intake from first narrative turn to final report export.

---

## 12.2 The 11 Test Modules (154 Passing Unit Tests)

The backend automated test suite (`backend/tests/`) contains exactly **11 test modules** comprising **154 automated unit and integration tests**:

```
+===================================================================================================+
|                                    ABHERA AUTOMATED TEST MATRIX                                   |
+----+-----------------------------------------+------------+---------------------------------------+
| #  | Test Module File                        | Test Count | Core Functionality Verified           |
+----+-----------------------------------------+------------+---------------------------------------+
| 01 | test_chat_api.py                        | 16 tests   | POST /api/chat/message, state machine |
| 02 | test_dynamic_question_generation.py     | 10 tests   | Contextual dynamic phrasing & slots   |
| 03 | test_fpm_incident.py                    | 12 tests   | False Promise of Marriage (BNS 69)    |
| 04 | test_hallucination.py                   | 13 tests   | Anti-Hallucination gating & provenance|
| 05 | test_legal_mapping.py                   | 9 tests    | PostgreSQL laws table normalization   |
| 06 | test_multilabel_classification.py       | 9 tests    | Multi-label probabilities & thresholds|
| 07 | test_question_engine.py                 | 18 tests   | Slot evaluation, rules, priority order|
| 08 | test_report_api.py                      | 20 tests   | POST & GET /api/report endpoints      |
| 09 | test_report_generator.py                | 22 tests   | 10-section structured dossier assembly|
| 10 | test_session_api.py                     | 13 tests   | POST & GET /api/session lifecycles    |
| 11 | test_support_mapping.py                 | 12 tests   | Emergency helpline 3-tier routing     |
+----+-----------------------------------------+------------+---------------------------------------+
|    | TOTAL AUTOMATED TEST SUITE              | 154 TESTS  | 100% PASSING RATE (0 Failures, 0 Reg) |
+===================================================================================================+
```

---

## 12.3 Test Suite Execution Command

To execute the complete 154-test suite on a Windows workstation, set `PYTHONPATH` to include the `backend/` directory and invoke Python's standard `unittest` test runner:

```powershell
# Windows PowerShell Test Runner Command
$env:PYTHONPATH="D:\Abhera(Mini)\backend;D:\Abhera(Mini)"
& "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m unittest discover -s backend/tests -v
```

*Prerequisite Note*: The PostgreSQL database service (`womens_safety_db`) must be active on `localhost:5432` during execution, as integration tests assert live database queries and foreign key constraints.

---

## 12.4 Detailed Breakdown of Key Test Suites

### 1. Anti-Hallucination & Provenance Tests (`test_hallucination.py` - 13 Tests)
Verifies that:
- Legitimate legal sections found in PostgreSQL pass validation with `True`.
- Injected hallucinated sections (e.g., `"IPC Section 999"`) fail validation with `False` and return descriptive error strings.
- Empty, `None`, or malformed dictionary payloads are safely intercepted.
- When no categories match, the mandatory safe fallback message (`LEGAL_NO_MATCH_MESSAGE`) is strictly returned.

### 2. Multi-Label Classification Tests (`test_multilabel_classification.py` - 9 Tests)
Verifies that:
- Single-category narratives (e.g., pure Stalking) output only `ST`.
- Compound narratives (e.g., physical assault by husband combined with threatening WhatsApp messages) simultaneously output both `DV` and `CA`.
- The optimal threshold tensor is loaded correctly and applied elementwise.
- Probabilities are bounded strictly within $[0.0, 1.0]$.

### 3. Dynamic Question Engine Tests (`test_question_engine.py` - 18 Tests)
Verifies that:
- When a narrative already contains an entity (e.g., *"my colleague"* $\implies$ `PERP_REL`), the engine marks that slot as known and does NOT ask for it.
- Follow-up questions are strictly ordered by canonical priority: `SAFETY_STATUS` (100) $\to$ `PERP_REL` (90) $\to$ `LOCATION` / `TIME_FREQ` / `PLATFORM` (85).
- When all required slots are satisfied, the engine transitions session status to `COMPLETED`.
- Conversational loops are mathematically prevented: an already-asked question ID is never re-selected.

### 4. Report Compilation Tests (`test_report_generator.py` - 22 Tests)
Verifies that:
- All 10 required conceptual sections exist in the returned dictionary.
- If a location was not mentioned in the initial narrative but was answered in a follow-up question, the generator correctly injects it into Section 3 (`extracted_information.LOCATION`).
- Section 9 correctly documents model names (`bert-base-uncased`, `bert-base-uncased-ner`) and data provenance.
- Section 10 includes the mandatory legal disclaimer.

---

## 12.5 Machine Learning Model Verification Results

### 1. BERT v2 Multi-Label Classification Benchmarks (164 Test Records)
- **Micro Precision**: 96.77%
- **Micro Recall**: 99.45%
- **Micro F1-Score**: 98.09%
- **Macro Precision**: 95.71%
- **Macro Recall**: 99.63%
- **Macro F1-Score**: 97.58%
- **Exact Match Accuracy**: 95.73%
- **Multi-Label Exact Match**: 94.12% (16 of 17 multi-label cases matched ground truth perfectly)
- **Elementwise Accuracy**: 99.29% (977 of 984 individual binary decisions correct)
- **Confusion Matrix**: True Positives = 180, False Positives = 6, False Negatives = 1, True Negatives = 797

### 2. BERT NER Token Classification Benchmarks (60 Test Records / 66 Entities)
- **Micro Precision**: 100.0% (Zero false positive hallucinations across all test records)
- **Micro Recall**: 98.48% (65 of 66 gold entity spans detected)
- **Micro F1-Score**: 99.24%
- **Per-Label Recall**:
  - `PERP_REL`: 100.0% (24 / 24 true spans)
  - `TIME_FREQ`: 100.0% (17 / 17 true spans)
  - `PLATFORM`: 100.0% (12 / 12 true spans)
  - `LOCATION`: 92.31% (12 / 13 true spans, 1 False Negative)

---

## 12.6 Known Issues and Engineering Limitations

To maintain academic integrity during examination, the following limitations are verified:
1. **PostgreSQL Dependency**: If the PostgreSQL service is offline, API tests connecting to the database fail with connection refused errors.
2. **CPU Inference Latency**: On standard quad-core laptop CPUs without CUDA hardware acceleration, running forward passes through both BERT models takes approximately 250ms to 450ms.
3. **Single Language Limitation**: The current production model operates strictly on English. Mixing Hindi or regional vernacular (Hinglish) produces lower classification confidence.
'''
