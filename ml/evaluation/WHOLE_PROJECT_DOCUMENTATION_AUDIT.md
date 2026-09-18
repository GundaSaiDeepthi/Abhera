# ABHERA — Whole Project Code Documentation & Comments Audit

> **Project:** ABHERA (AI-Powered Legal & Support Assistant for Women's Safety)  
> **Audit Type:** Whole-Project Code Documentation & Comment Audit  
> **Date:** September 18, 2026  
> **Auditor:** Automated Engineering Documentation Audit Pipeline  
> **Documentation Pass Outcome:** PASS  

---

## 1. Files Inspected

The entire ABHERA workspace codebase was systematically inspected across 128 source files spanning backend APIs, machine learning pipelines, evaluation benchmarks, database models, and frontend components:

- **Backend Python Modules:** 57 files (`backend/app/**/*.py`, `backend/tests/*.py`)
- **ML Training, Inference & Evaluation Modules:** 37 files (`ml/**/*.py`)
- **Frontend Source Files:** 19 files (`frontend/src/**/*.{jsx,js,css}`)
- **Database & Script Utilities:** Init, migration, seed, and data import scripts.

---

## 2. Files Documented

Comprehensive professional software-engineering docstrings and comments were added or enhanced across all core project layers:

- **ML Inference:**
  - `ml/inference/predict_bert.py` (BERT Multi-Label Predictor, Sigmoid activation, validation thresholds, rule-based evidence reinforcement)
  - `ml/inference/predict_ner.py` (BERT Token Classification, BIO tagging scheme, sub-token offset alignment)
- **Backend Core & Database:**
  - `backend/app/database.py` (SQLAlchemy engine, connection pooling `pool_pre_ping=True`, declarative base, session dependency)
  - `backend/app/main.py` (FastAPI application initialization, CORS middleware, global exception handlers)
  - `backend/app/config.py` (Environment variables, database URL resolution, JWT/Secret keys handling)
- **Backend Routes & Services:**
  - `backend/app/routes/chat.py` (Conversational REST endpoint `/api/chat/message`, payload validation, error contracts)
  - `backend/app/routes/session.py` (Session creation `/api/session/start` and retrieval endpoints)
  - `backend/app/routes/report.py` (Incident report generation `/api/report/generate` and retrieval endpoints)
  - `backend/app/services/chat_service.py` (Central chat orchestrator, lazy predictor loading, multi-stage pipeline flow)
  - `backend/app/services/session_service.py` (Session state lifecycle & persistence)
  - `backend/app/services/report_generator.py` (10-Section legal incident summary compiler)
- **Database ORM Entities:**
  - `backend/app/models/user.py`, `session.py`, `incident_submission.py`, `conversation_message.py`, `prediction.py`, `entity.py`, `question.py`, `law.py`, `support_service.py`, `report.py`
- **Frontend State & Services:**
  - `frontend/src/context/ConversationContext.jsx` (Global React Context, `sessionStorage` sync, state management)
  - `frontend/src/services/api.js` (Axios HTTP client, endpoint definitions, environment URL fallbacks)
  - `frontend/src/pages/Chatbot.jsx`, `Home.jsx`, `Processing.jsx`, `Report.jsx`
- **Evaluation & Benchmark Package:**
  - `ml/evaluation/generate_paper_evaluation_package.py`, `FINAL_EVALUATION_REPORT.md`, `PAPER_RESULTS_SECTION.md`, `FINAL_PAPER_CONSISTENCY_CHECK.md`.

---

## 3. Major Modules Documented

Each major module contains detailed top-level docstrings explaining:
1. **Purpose:** Primary architectural role and functional responsibilities.
2. **Why:** Rationale behind critical design decisions (e.g. multi-label Sigmoids over Softmax, validation threshold search over default 0.50, PostgreSQL/SQLite ORM mapping over raw SQL).
3. **Input/Output:** Structure of input arguments, payloads, and return objects.
4. **Non-Obvious Logic:** Detailed explanations of multi-label fallback heuristics, sub-token alignment loops, anti-hallucination guardrails, and dynamic question deduplication.

---

## 4. BERT Documentation Coverage

The fine-tuned BERT classification pipeline is fully documented across preprocessing, training, threshold tuning, and inference:
- **Architecture:** `bert-base-uncased` sequence classification head (110M parameters).
- **Multi-Label Formulation:** Independent Sigmoid outputs ($p_i = 1 / (1 + e^{-z_i})$) allowing multiple concurrent incident tags (e.g. Domestic Violence + Cyber Harassment).
- **Threshold Calibration:** Validation-derived optimal decision thresholds (`DV`: 0.73, `SH`: 0.50, `ST`: 0.79, `CA`: 0.45, `WH`: 0.29, `OV`: 0.63) tuned on validation set ($N=162$) to optimize Micro-F1 without exposing held-out test data.
- **Evidence Reinforcement:** Explicit rules enforcing safety overlays for physical domestic abuse, dowry harassment, and false promise of marriage contexts.

---

## 5. NER Documentation Coverage

The Named Entity Recognition pipeline is comprehensively documented:
- **Architecture:** `bert-base-uncased` token-level classification head.
- **Tagging Scheme:** BIO (Begin, Inside, Outside) tags for `PERP_REL`, `LOCATION`, `DATE_TIME`, `CONTACT_INFO`, and `EVIDENCE`.
- **Sub-token Alignment:** Offset mapping conversion from WordPiece sub-tokens (`##ing`) to raw string character offsets (`start_char`, `end_char`).
- **Functional Separation:** Clearly distinguishes token-level NER span extraction from sequence-level BERT incident classification.

---

## 6. Backend Documentation Coverage

FastAPI routers, schemas, and service orchestrators are thoroughly documented:
- **Routing:** Contract specifications for `/api/session`, `/api/chat`, `/api/report`, `/health`.
- **End-to-End Chat Pipeline:** User message $\rightarrow$ Session lookup $\rightarrow$ Question answer capture $\rightarrow$ BERT prediction $\rightarrow$ NER extraction $\rightarrow$ Question Engine selection $\rightarrow$ Statutory Law/Support lookup $\rightarrow$ Anti-hallucination guard $\rightarrow$ Markdown report compilation $\rightarrow$ DB commit.
- **Safeguards:** Explicit documentation of HTTP error handling (400, 404, 500) and graceful predictor fallback mechanisms.

---

## 7. Database Documentation Coverage

All SQLAlchemy ORM database entities and relationships are documented:
- **Entity Purpose:** Persistence of Users, Sessions, Incident Submissions, Messages, Predictions, Extracted Entities, Questions, Answers, IPC/BNS Laws, Helplines, and Reports.
- **Relational Integrity:** Foreign key constraints (`submission_id`, `session_id`, `user_id`), cascading deletions, and query indexing strategies.

---

## 8. Frontend Documentation Coverage

React components, hooks, service layers, and state management are documented:
- **State Flow:** React Context (`ConversationContext.jsx`) managing intake lifecycle, session persistence, message streams, and ML predictions.
- **API Service:** Axios instance (`services/api.js`) handling base URL configuration, headers, and request/response interceptors.
- **UI Components:** Chatbot interaction (`Chatbot.jsx`), Processing indicators (`Processing.jsx`), Markdown report renderer (`Report.jsx`), and jsPDF export trigger.

---

## 9. Evaluation Documentation Coverage

Evaluation scripts and benchmark generation packages are documented:
- **Metrics Covered:** Exact Match Accuracy (95.73%), Elementwise Accuracy (99.29%), Micro Precision (96.77%), Micro Recall (99.45%), Micro F1-Score (98.09%), Macro F1-Score (97.58%), entity-level Micro-F1 (99.24%).
- **Evaluation Methodology:** Multi-label subset disaggregation, per-label confusion matrices, threshold isolation, and 154-test regression suites.

---

## 10. Question Engine Documentation Coverage

The Dynamic Question Engine (`backend/app/question_engine/`) is documented:
- **Adaptive Selection:** Information gap analysis identifying missing entity categories (e.g. Perpetrator, Location, Evidence).
- **Deduplication:** Tracking asked question IDs to prevent duplicate prompts during intake.
- **Rules & Triggers:** Category-specific follow-up question lookup and completion threshold verification.

---

## 11. Legal / Support Mapping Documentation Coverage

Legal and Support service mapping services are documented:
- **DB-Backed Retrieval:** Category-indexed mapping retrieving verified Indian Penal Code (IPC), Bharatiya Nyaya Sanhita (BNS), PWDVA, POSH, and IT Act legal sections.
- **Helpline Matching:** Verification of official helpline contacts (181 Women Helpline, 112 ERSS, 1930 Cyber Helpline, One Stop Centres).
- **Anti-Hallucination Guard:** Fallback logic ensuring unmapped or low-confidence queries return verified static advice rather than fabricated statutory information.

---

## 12. Report / PDF Documentation Coverage

Report assembly and document generation modules are documented:
- **Executive Summary Assembly:** Markdown engine compiling 10 structured conceptual sections (Incident Summary, Identified Incident Types, Extracted Details, Statutory Legal Info, Next Steps, Support Services, Disclaimer).
- **PDF Export Pipeline:** Client-side jsPDF and html2pdf rendering workflow.

---

## 13. Security / Privacy Documentation Coverage

Security and survivor privacy design patterns are documented:
- **Credential Hygiene:** Zero hardcoded API keys, passwords, or tokens in comments or codebase.
- **Session Isolation:** Ephemeral session tokens stored in `sessionStorage` rather than persistent cookies.
- **Data Protection:** Database anonymization guidelines for victim narrative inputs.

---

## 14. Tests Executed

Automated unit and integration tests were executed to verify zero functional regressions:
- **Backend PyTest / Unittest Suite:** Executed cleanly across all test files (`test_chat_api.py`, `test_dynamic_question_generation.py`, `test_fpm_incident.py`, `test_hallucination.py`, `test_legal_mapping.py`, `test_question_engine.py`, `test_report_api.py`, `test_report_generator.py`, `test_session_api.py`, `test_support_mapping.py`).
- **Regression Pass Rate:** **154 / 154 Passed (100.0%)**.

---

## 15. Frontend Build Result

The production frontend build was executed via Vite / React:
- **Command:** `npm run build` inside `frontend/`
- **Build Status:** SUCCESS (0 compilation errors, 0 asset bundle errors).

---

## 16. Files Intentionally Left Unchanged

The following files were intentionally preserved without comment modification:
- **Binary & Artifact Files:** PyTorch model weights (`.bin`, `.safetensors`), tokenizers (`vocab.txt`), database files (`.db`, `.sqlite`).
- **Machine-Generated CSV/JSON Outputs:** Evaluation metrics JSON/CSV files (comments not supported in standard JSON/CSV format).
- **Node Modules & Virtual Envs:** Third-party dependency packages (`node_modules/`, `venv/`).

---

## 17. Confirmation of Zero Code & Logic Mutation

It is explicitly confirmed that:
1. **0 lines of business logic were modified.**
2. **0 ML model parameters, thresholds, weights, or datasets were changed.**
3. **0 API endpoint signatures, database schemas, or legal mappings were altered.**
4. **0 frontend UI components or styling definitions were mutated.**
5. **The documentation pass consisted strictly of adding and enhancing high-quality comments, docstrings, and markdown documentation.**

---

WHOLE PROJECT DOCUMENTATION STATUS: PASS
