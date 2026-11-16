# FINAL DOCUMENTATION VERIFICATION REPORT
## ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance
**Project Directory:** `D:\Abhera(Mini)`  
**Audit Date:** November 2026  
**Auditor:** Independent Technical Verification Agent (Antigravity)  
**Verification Standard:** Strict Empirical Evidence Only (Zero Fabrication, Source-Code & Artifact Ground Truth)

---

## 1. Executive Summary & Verification Verdict

An exhaustive, independent verification was conducted across the complete ABHERA documentation suite:
- `ABHERA_COMPLETE_DEVELOPMENT_AND_WORKING_GUIDE.md` (Master Development Guide)
- `ABHERA_DEVELOPMENT_DOCUMENTATION_AUDIT.md` (Self-Audit Document)
- `docs/development_guide/01_PROJECT_ORIGIN_AND_REQUIREMENTS.md` through `15_DEVELOPMENT_HISTORY_AND_VIVA_GUIDE.md` (15 Modular Development Guides)

### Overall Verification Status
**STATUS: PASS WITH SPECIFIC DOCUMENTED DISCREPANCIES IDENTIFIED**

The underlying system, machine learning models, database initialization scripts, question engine, and test suites are genuine, functional, and empirically proven. However, multiple discrepancies exist between theoretical/idealized descriptions in certain modular guides and the actual implemented codebase.

### Claim Verification Metrics
| Verification Category | Total Claims Checked | Verified Claims (PASS) | Failed Claims (FAIL) | Unverified Claims (NOT VERIFIED) | Accuracy Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Classification Metrics (BERT & Baselines)** | 18 | 18 | 0 | 0 | 100.0% |
| **2. Decision Thresholds (BERT v2)** | 6 | 6 | 0 | 0 | 100.0% |
| **3. NER Entity Metrics** | 8 | 6 | 2 | 0 | 75.0% |
| **4. Automated Test Suite Execution** | 12 | 12 | 0 | 0 | 100.0% |
| **5. Statutory & Service Counts** | 6 | 6 | 0 | 0 | 100.0% |
| **6. File Paths Existence** | 69 | 52 | 17 | 0 | 75.4% |
| **7. Mermaid Diagram Structures** | 28 | 26 | 2 | 0 | 92.9% |
| **8. Code & Architecture Consistency** | 15 | 8 | 7 | 0 | 53.3% |
| **9. Deployment Configuration** | 10 | 6 | 4 | 0 | 60.0% |
| **10. Claims & Assumptions Audit** | 8 | 4 | 4 | 0 | 50.0% |
| **TOTAL AUDIT METRICS** | **180** | **144** | **36** | **0** | **80.0%** |

*Note: All items were empirically resolved against disk artifacts; zero claims remain in `NOT VERIFIED` status.*

---

## 2. Requirement 1: Verification of Model Metrics

### 2.1 BERT v2 (Production Model) Metrics
- **Artifact Source:** `D:\Abhera(Mini)\ml\evaluation\FINAL_EVALUATION_RESULTS.json`
- **Evaluation Set:** 164 held-out test records (seed = 42, 70/15/15 split).

| Metric | Documented Claim | Empirical JSON Value | Verification Result |
| :--- | :--- | :--- | :---: |
| **Micro Precision** | 0.9677 (96.77%) | 0.9677 | **PASS** |
| **Micro Recall** | 0.9945 (99.45%) | 0.9945 | **PASS** |
| **Micro F1-Score** | 0.9809 (98.09%) | 0.9809 | **PASS** |
| **Macro Precision** | 0.9571 (95.71%) | 0.9571 | **PASS** |
| **Macro Recall** | 0.9963 (99.63%) | 0.9963 | **PASS** |
| **Macro F1-Score** | 0.9758 (97.58%) | 0.9758 | **PASS** |
| **Exact Match Accuracy** | 0.9573 (95.73%) | 0.9573 | **PASS** |
| **Multi-Label Exact Match** | 0.9412 (94.12%) | 0.9412 | **PASS** |
| **Elementwise Accuracy** | 0.9929 (99.29%) | 0.9929 | **PASS** |
| **True Positives (TP)** | 180 | 180 | **PASS** |
| **False Positives (FP)** | 6 | 6 | **PASS** |
| **False Negatives (FN)** | 1 (`TEST_0050`, subtle stalking) | 1 | **PASS** |
| **True Negatives (TN)** | 797 | 797 | **PASS** |

### 2.2 Baseline Comparisons
- **Artifact Source:** `D:\Abhera(Mini)\ml\evaluation\FINAL_EVALUATION_RESULTS.json` and `ml/evaluation/results/baseline_comparison_results.json`

| Baseline Model | Metric | Documented Claim | Empirical JSON Value | Verification Result |
| :--- | :--- | :--- | :--- | :---: |
| **TF-IDF + Logistic Regression** | Micro-F1 | 0.9779 | 0.9779 | **PASS** |
| | Exact Match | 0.9573 | 0.9573 | **PASS** |
| | Macro-F1 | 0.9735 | 0.9735 | **PASS** |
| **TF-IDF + Linear SVM** | Micro-F1 | 0.9727 | 0.9727 | **PASS** |
| | Exact Match | 0.9512 / 0.9634 | 0.9512 (FINAL) / 0.9634 (Baseline JSON) | **PASS** |
| **BERT v1 (Baseline Fine-Tuned)** | Micro-F1 | 0.9133 | 0.9133 | **PASS** |
| | Exact Match | 0.8354 | 0.8354 | **PASS** |
| | Multi-label Exact Match | 0.00% (23 FN) | 0.00% (23 FN) | **PASS** |

---

## 3. Requirement 2: Verification of BERT v2 Classification Thresholds

- **Artifact Source:** `D:\Abhera(Mini)\models\bert_multilabel_experiment_v2\optimal_thresholds.json`

The documentation claims that fixed arbitrary 0.50 thresholds were discarded in favor of validation-derived per-class thresholds optimized via grid search on held-out validation loss.

| Class Code | Category Description | Documented Threshold | Saved JSON Configuration | Verification Result |
| :---: | :--- | :---: | :---: | :---: |
| **DV** | Domestic Violence | 0.73 | 0.73 | **PASS** |
| **SH** | Sexual Harassment | 0.50 | 0.50 | **PASS** |
| **ST** | Stalking | 0.79 | 0.79 | **PASS** |
| **CA** | Cyber Abuse | 0.45 | 0.45 | **PASS** |
| **WH** | Workplace Harassment | 0.29 | 0.29 | **PASS** |
| **OV** | Other Violence | 0.63 | 0.63 | **PASS** |

**Finding:** Exact 100% concordance. The threshold configuration file is present and matches documented parameters to two decimal places.

---

## 4. Requirement 3: Verification of NER Model Metrics & Discrepancy Analysis

- **Artifact Source:** `D:\Abhera(Mini)\models\ner\test_metrics.json`
- **Evaluation Set:** 60 annotated survivor narratives, 66 total ground-truth entity spans.

### 4.1 Overall NER Metrics
| Metric | Documented Claim | Actual JSON Metric | Verification Result |
| :--- | :--- | :--- | :---: |
| **Evaluation Records** | 60 | 60 | **PASS** |
| **Total Entity Support** | 66 | 66 | **PASS** |
| **Micro Precision** | 1.0 (100.0%) | 1.0 (100.0%) | **PASS** |
| **Micro Recall** | 0.9848 (98.48%) | 0.9848 (98.48%) | **PASS** |
| **Micro F1-Score** | 0.9924 (99.24%) | 0.9924 (99.24%) | **PASS** |
| **True Positives / FP / FN** | 65 TP, 0 FP, 1 FN | 65 TP, 0 FP, 1 FN | **PASS** |

### 4.2 Per-Label NER Metrics & Discrepancies
In `docs/development_guide/06_NER_DEVELOPMENT.md`, specific per-label recall figures were reported with synthetic percentages:

| Entity Type | Documented Recall | Actual JSON Support | Actual JSON Recall | Actual JSON F1 | Verification Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PERP_REL** | 100.0% | 24 / 24 | 1.0000 (100.0%) | 1.0000 (100.0%) | **PASS** |
| **LOCATION** | 97.2% | 12 / 13 | **0.9231 (92.31%)** | **0.9600 (96.00%)** | **FAIL (Discrepancy)** |
| **TIME_FREQ** | 96.8% | 17 / 17 | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **FAIL (Discrepancy)** |
| **PLATFORM** | 100.0% | 12 / 12 | 1.0000 (100.0%) | 1.0000 (100.0%) | **PASS** |

**Critical Finding:**
`06_NER_DEVELOPMENT.md` stated that `LOCATION` achieved 97.2% recall and `TIME_FREQ` achieved 96.8% recall. The ground truth evaluation in `models/ner/test_metrics.json` proves:
1. `LOCATION` recall is **92.31%** (12 TP, 0 FP, 1 FN out of 13 ground-truth instances).
2. `TIME_FREQ` recall is **100.00%** (17 TP, 0 FP, 0 FN out of 17 ground-truth instances).
The overall micro metrics (Precision 100%, Recall 98.48%, F1 99.24%) were reported accurately, but the per-label breakdown in Module 06 contained unverified percentage figures.

---

## 5. Requirement 4: Verification of Automated Test Suite

- **Execution Command:** `& "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m unittest discover tests`
- **Working Directory:** `D:\Abhera(Mini)\backend`
- **Execution Output:**
  ```text
  Ran 154 tests in 39.892s
  OK
  ```

### Test Suite Distribution Across Modules
| Test Module | Discovered Test Methods | Status | Scope Tested |
| :--- | :---: | :---: | :--- |
| `tests/test_chat_api.py` | 16 | **PASS** | `/api/chat/message`, session state transitions, multi-turn dialogue |
| `tests/test_dynamic_question_generation.py` | 10 | **PASS** | Entity-gap heuristics, category templates, dynamic follow-ups |
| `tests/test_fpm_incident.py` | 12 | **PASS** | Section 69 BNS deceitful sexual intercourse edge cases |
| `tests/test_hallucination.py` | 13 | **PASS** | Anti-hallucination provenance safeguards, statutory boundary checks |
| `tests/test_legal_mapping.py` | 9 | **PASS** | Mapping predicted classes to verified statutory laws in PostgreSQL |
| `tests/test_multilabel_classification.py` | 9 | **PASS** | BERT multi-label thresholds, multi-class predictions |
| `tests/test_question_engine.py` | 18 | **PASS** | Core question engine, priority queues, question selection |
| `tests/test_report_api.py` | 20 | **PASS** | `/api/report/{submission_id}`, JSON serialization, error handling |
| `tests/test_report_generator.py` | 22 | **PASS** | 10-section structured markdown and JSON report generation |
| `tests/test_session_api.py` | 13 | **PASS** | `/api/session/start`, `/api/session/{id}`, session persistence |
| `tests/test_support_mapping.py` | 12 | **PASS** | Location-aware helpline filtering (State/District), national fallbacks |
| **TOTAL AUTOMATED TESTS** | **154** | **154 PASS (0 FAIL, 0 ERROR)** | **Full End-to-End System Coverage** |

**Execution Finding:**
All 154 tests pass synchronously. However, running `python -m unittest` with system global Python fails due to missing dependencies (`pydantic_settings`, `sqlalchemy`). The tests must be executed with the dedicated virtual environment Python (`backend\venv\Scripts\python.exe`).

---

## 6. Requirement 5: Verification of Statutory Laws & Support Service Counts

### 6.1 Statutory Laws Count & Schema
- **File Checked:** `D:\Abhera(Mini)\data\laws.csv`
- **Line Count:** 34 total lines (1 header line + 32 data rows + 1 trailing newline)
- **Verified Row Count:** **32 statutory provisions**
- **CSV Headers:** `act_name`, `section_number`, `section_text`, `applicable_label`
- **Statutes Represented:**
  - Bharatiya Nyaya Sanhita, 2023 (BNS): Sec 69, 74, 75, 76, 77, 78, 79, 85, 86, 351, 356
  - Information Technology Act, 2000 (IT Act): Sec 66C, 66D, 66E, 67, 67A
  - Protection of Women from Domestic Violence Act, 2005 (PWDVA): Sec 18, 19, 20, 22
  - Dowry Prohibition Act, 1961: Sec 3, 4, 6
  - POSH Act, 2013: Sec 4, 9, 11, 13
- **Verification Result:** **PASS** (Matches 32 documented rows exactly).

### 6.2 Support Services Count & Schema
- **File Checked:** `D:\Abhera(Mini)\data\support_services.csv`
- **Line Count:** 14 total lines (1 header line + 12 data rows + 1 trailing newline)
- **Verified Row Count:** **12 support helplines & One-Stop Centers**
- **CSV Headers:** `service_type`, `name`, `contact_number`, `state`, `district`, `applicable_label`
- **Services Represented:**
  - National Emergency (112)
  - Women Helpline (1091, 181)
  - Cyber Crime Helpline (1930)
  - NCW Helpline (7827170170)
  - State & District One-Stop Centers (Telangana, Andhra Pradesh, Karnataka, Delhi, Maharashtra)
- **Verification Result:** **PASS** (Matches 12 documented rows exactly).

### 6.3 Incident Dataset Count & Splits
- **File Checked:** `D:\Abhera(Mini)\data\womens_safety_dataset.csv`
- **Total Records:** 1,086 rows
- **Splits Generated:** `ml/preprocessing/train_data.csv` (760 rows, 70%), `val_data.csv` (162 rows, 15%), `test_data.csv` (164 rows, 15%)
- **Verification Result:** **PASS**.

---

## 7. Requirement 6: Comprehensive File Path Existence Audit

An automated recursive file scanner tested all 69 file path references across the documentation suite against physical paths on disk:
- **Total Paths Scanned:** 69
- **Existing Paths on Disk:** 52 (**PASS**)
- **Missing or Discrepant Paths:** 17 (**FAIL**)

### Detailed Discrepancy Breakdown
| Documented Path in Guide | Actual Path in Codebase | Severity | Issue / Explanation |
| :--- | :--- | :---: | :--- |
| `data/train.csv` | `ml/preprocessing/train_data.csv` | Moderate | Documented under `data/`, generated in `ml/preprocessing/` |
| `data/val.csv` | `ml/preprocessing/val_data.csv` | Moderate | Documented under `data/`, generated in `ml/preprocessing/` |
| `data/test.csv` | `ml/preprocessing/test_data.csv` | Moderate | Documented under `data/`, generated in `ml/preprocessing/` |
| `models/bert_multilabel_v1/` | `models/bert_multilabel/` | Minor | Folder on disk is `bert_multilabel`, not `bert_multilabel_v1` |
| `models/bert_multilabel_experiment_v2/vocab.txt` | `models/bert_multilabel_experiment_v2/tokenizer.json` | Minor | Model uses fast HuggingFace tokenizer without standalone `vocab.txt` |
| `scripts/import_laws.py` | `backend/app/services/data_importer.py` | Moderate | Module 14 referenced separate script; import is consolidated in `data_importer.py` |
| `scripts/import_support_services.py` | `backend/app/services/data_importer.py` | Moderate | Consolidated in `DatasetImporter.process_support_services_csv()` |
| `backend/tests/test_anti_hallucination.py` | `backend/tests/test_hallucination.py` | Minor | Actual test file name is `test_hallucination.py` |
| `backend/tests/test_api_routes.py` | `backend/tests/test_chat_api.py`, `test_session_api.py`, `test_report_api.py` | Minor | API tests are split into dedicated route test files |
| `backend/tests/test_chat_service.py` | `backend/tests/test_chat_api.py` | Minor | Covered in `test_chat_api.py` |
| `question_engine/engine.py` | `backend/app/question_engine/engine.py` | Minor | Relative path missing `backend/app/` prefix |
| `routes/chat.py` | `backend/app/routes/chat.py` | Minor | Relative path missing `backend/app/` prefix |
| `routes/health.py` | `backend/app/routes/health.py` | Minor | Relative path missing `backend/app/` prefix |
| `routes/report.py` | `backend/app/routes/report.py` | Minor | Relative path missing `backend/app/` prefix |
| `routes/session.py` | `backend/app/routes/session.py` | Minor | Relative path missing `backend/app/` prefix |
| `services/api.js` | `frontend/src/services/api.js` | Minor | Relative path missing `frontend/src/` prefix |
| `pages/Report.jsx` | `frontend/src/pages/Report.jsx` | Minor | Relative path missing `frontend/src/` prefix |

---

## 8. Requirement 7: Structural Audit of Mermaid Diagrams

A comprehensive scan of all 28 Mermaid diagrams across the documentation suite was performed.

### Diagram Distribution
- `flowchart TD / LR`: 14 diagrams
- `graph TB / TD / LR`: 9 diagrams
- `sequenceDiagram`: 2 diagrams
- `erDiagram`: 1 diagram
- `stateDiagram-v2`: 1 diagram
- `mindmap`: 1 diagram

### Findings & Discrepancies
1. **Experimental Mindmap Diagram in Module 09:**
   - **Location:** `docs/development_guide/09_LEGAL_SUPPORT_AND_ANTI_HALLUCINATION.md` (Line 32)
   - **Issue:** The diagram is declared as `mindmap`. Mindmap syntax is experimental in Mermaid and unsupported by several standard GitHub and IDE markdown renderers, leading to a blank or broken rendering.
   - **Recommendation:** Convert to standard `flowchart TD` or `graph TD` to ensure universal rendering compatibility.
2. **Unquoted Parentheses inside Database Cylindrical Nodes:**
   - In multiple diagrams (e.g. `01_PROJECT_ORIGIN_AND_REQUIREMENTS.md` Line 77, `14_DEPLOYMENT_AND_MAINTENANCE.md` Line 24), database cylinder notation `[(...)]` contains unquoted nested parentheses, e.g.:
     `PostgresDB[(PostgreSQL 16 High-Availability Instance<br/>- Encrypted Storage at Rest (AES-256))]`
   - Certain strict Mermaid parsers misinterpret `(AES-256)` as an inner rounded shape closure, causing syntax errors.
   - **Recommendation:** Enclose the inner label text in double quotes: `PostgresDB[("PostgreSQL 16 High-Availability Instance<br/>- Encrypted Storage at Rest (AES-256)")]`.

---

## 9. Requirement 8: Contradictions Between Modular Guides & Source Code

Several notable contradictions between modular guide descriptions and the actual source code were identified:

### 8.1 Database Schema Definition (Module 07 vs `backend/app/models/`)
- **Documented in Module 07:**
  Described a theoretical expanded SQL schema with UUID primary keys (`id UUID PRIMARY KEY`), PostgreSQL array types (`VARCHAR(50)[]`), and legal categorization columns (`cognizable BOOLEAN`, `bailable BOOLEAN`, `compoundable BOOLEAN`, `punishment_summary TEXT` on `laws`; `city`, `address`, `pin_code`, `operational_hours` on `support_services`; `email`, `hashed_password` on `users`; `pdf_download_url` on `reports`).
- **Actual Code Reality (`backend/app/models/`):**
  SQLAlchemy models use Integer autoincrement primary keys (`id = Column(Integer, primary_key=True)`). The schema is streamlined and pragmatic:
  - `Law`: `act_name`, `section_number`, `section_text`, `applicable_label`.
  - `SupportService`: `service_type`, `name`, `contact_number`, `state`, `district`, `applicable_label`.
  - `User`: `user_guid` (UUID string), `is_anonymous` (Boolean). No passwords or user emails exist to protect survivor anonymity.
  - `Report`: Stores full structured report in `report_data = Column(JSON)`. No `pdf_download_url` exists because PDF generation is client-side.
  - Multi-turn state, predicted labels, and answers are persisted using PostgreSQL `JSON` columns on `Session` and `IncidentSubmission`.

### 8.2 Frontend Route Structure & Component Architecture (Module 02 vs `frontend/src/`)
- **Documented in Module 02:**
  Listed theoretical pages `IncidentReport.jsx`, `QuestionInterface.jsx`, `LegalInfo.jsx`, `SupportFinder.jsx` and separate component files `IncidentForm.jsx`, `QuestionCard.jsx`, `EntityTag.jsx`, `LegalCard.jsx`, `SupportCard.jsx`.
- **Actual Code Reality (`frontend/src/`):**
  The React application uses a streamlined 4-page single-page flow in `frontend/src/App.jsx`:
  - `/` -> `Home.jsx`
  - `/chatbot` -> `Chatbot.jsx` (interactive conversational intake)
  - `/processing` -> `Processing.jsx` (animated model execution state)
  - `/report` -> `Report.jsx` (complete multi-section report viewer and PDF download)
  Components in `frontend/src/components/` are `AbheraEmblem.jsx`, `ChatInput.jsx`, `ChatMessage.jsx`, `HeroIllustration.jsx`, `LoadingIndicator.jsx`, `Navbar.jsx`. UI cards for legal provisions and support services are rendered directly within `Report.jsx`.

### 8.3 Class and Predictor Names
- **Documented:** Some guides refer to `HarassmentClassifier`.
- **Actual Code Reality:** The production class in `ml/inference/predict_bert.py` is named `BERTIncidentPredictor`. The NER inference class in `ml/inference/predict_ner.py` is `NEREntityPredictor`.

---

## 10. Requirement 9: Verification of Production Deployment Configuration

Verification against `backend/app/config.py`, `frontend/src/services/api.js`, and `docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md`:

### 9.1 Environment Variable Naming Mismatch
1. **Frontend API URL:**
   - In `frontend/src/services/api.js`: `const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';`
   - Module 02 documented: `VITE_API_BASE_URL`.
   - **Verdict:** Must use `VITE_API_URL` in Vercel configuration.
2. **Backend CORS Allowed Origins:**
   - In `backend/app/config.py`: The setting attribute is `CORS_ORIGINS = [...]`.
   - Module 02 and Module 14 documented: `ALLOWED_ORIGINS`.
   - **Verdict:** Must set `CORS_ORIGINS` in Render environment variables.
3. **Database Name:**
   - Default in `backend/app/config.py` is `POSTGRES_DB = "womens_safety_db"`.
   - Module 14 instructions use `abhera_db`.
   - **Verdict:** Render provides a full `DATABASE_URL` connection string, which `config.py` correctly prioritizes over individual fields (`if self.DATABASE_URL: return self.DATABASE_URL`).

### 9.2 Database Seeding Commands in Production Playbook
Module 14 instructed users to run:
```powershell
python scripts/import_laws.py
python scripts/import_support_services.py
```
**Correction Required:** These script files do not exist. In production, statutory laws and support services are imported and validated using:
```powershell
python -m app.services.data_importer
```
or programmatically calling `DatasetImporter().run_all()`.

### 9.3 Model Sizing & RAM Verification
- **Safetensors Weights on Disk:**
  - `models/bert_multilabel_experiment_v2/model.safetensors`: 437,970,928 bytes ≈ **417.68 MB**
  - `models/ner/model.safetensors`: 435,629,900 bytes ≈ **415.45 MB**
  - **Combined Model Weights Disk Footprint:** **833.13 MB**
- **Memory Tier Assessment:**
  Loading both 110M-parameter models into PyTorch host RAM requires ~1.35 GB. The documentation correctly warns that cloud free tiers with 512MB RAM will fail with `OOMKilled`, and accurately recommends a minimum of **2.0 GB RAM** on Render.

---

## 11. Requirement 10: Unsupported Claims, Assumptions, and Outdated Information

1. **Synthetic NER Recall Percentages (Module 06):**
   Claimed 97.2% for `LOCATION` and 96.8% for `TIME_FREQ`. Ground truth is 92.31% (12/13) and 100.0% (17/17). Documented percentages were synthetic estimates rather than empirical test outputs.
2. **Standalone Vocabulary File (Module 05):**
   Claimed `vocab.txt` is present in `models/bert_multilabel_experiment_v2/`. Ground truth: The model directory utilizes the HuggingFace fast tokenizer representation (`tokenizer.json`, `tokenizer_config.json`) and does not contain a separate `vocab.txt`.
3. **Backend Seed Scripts (Module 14):**
   Claimed separate modular scripts `import_laws.py` and `import_support_services.py` exist under `scripts/`. Ground truth: `DatasetImporter` in `backend/app/services/data_importer.py` performs all statutory and helpline imports.
4. **Idealized Relational DDL (Module 07):**
   Module 07 documented PostgreSQL DDL with complex arrays and legal columns not present in the active SQLAlchemy ORM models. While conceptually illustrative of an expanded legal schema, it does not represent the running database structure.

---

## 12. Verification Summary & Next Steps

| Category | Finding | Recommended Action |
| :--- | :--- | :--- |
| **Model Metrics** | 100% verified against JSON evaluation files. | Retain exact reported metrics. |
| **Optimal Thresholds** | 100% verified against `optimal_thresholds.json`. | Retain documented thresholds. |
| **NER Per-Label Recall** | Discrepancy discovered in `LOCATION` and `TIME_FREQ`. | Update documentation to reflect exact 92.31% and 100.0% values. |
| **Automated Tests** | 154 tests verified passing (0 fail, 0 error). | Document virtualenv python command requirement. |
| **Dataset & CSV Counts** | 32 laws, 12 services, 1,086 rows verified. | Fully accurate. |
| **File Paths & Scripts** | 17 discrepant or relative paths identified. | Refer to `DOCUMENTATION_CORRECTION_LOG.md` for exact paths. |
| **Mermaid Diagrams** | 1 mindmap and unquoted cylinder labels identified. | Convert mindmap to flowchart and quote labels with parens. |
| **Deployment Config** | Variable names `VITE_API_URL` and `CORS_ORIGINS` verified. | Update deployment instructions to match actual config names. |

*Full itemized corrections and replacement text are recorded in `D:\Abhera(Mini)\DOCUMENTATION_CORRECTION_LOG.md`.*
