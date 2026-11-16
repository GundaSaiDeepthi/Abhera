# DOCUMENTATION CORRECTION LOG
## ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance
**Project Directory:** `D:\Abhera(Mini)`  
**Audit Date:** November 2026  
**Log Status:** Final Independent Verification Complete  
**Purpose:** Precise item-by-item record of documented statements vs. empirical codebase reality, providing exact replacement text for viva defense, technical paper submission, and project maintenance.

---

## 1. Summary of Discrepancies by Severity

| Severity Level | Count | Description / Impact |
| :---: | :---: | :--- |
| **CRITICAL** | 2 | Discrepancies involving empirical evaluation metrics or non-functional deployment commands that would cause viva failure or production crash. |
| **MODERATE** | 5 | Architectural discrepancies where theoretical schemas or file paths differ from actual implemented components. |
| **MINOR** | 5 | Syntactic warnings, relative path prefixes, environment variable aliases, or renderer compatibility notices. |
| **TOTAL** | **12** | **Full Audit Resolution Complete** |

---

## 2. Item-by-Item Correction Records

### CORR-001: NER Per-Label Evaluation Recall Discrepancy
- **Target Files:**
  - `docs/development_guide/06_NER_DEVELOPMENT.md` (Section 5, Evaluation Results)
  - `ABHERA_COMPLETE_DEVELOPMENT_AND_WORKING_GUIDE.md` (Section 5)
- **Severity:** **CRITICAL**
- **Documented Statement:**
  > "Evaluation on the 60-sample held-out NER test set yielded 100.0% precision, 98.48% recall, and 99.24% F1. Per-label recall: PERP_REL: 100%, LOCATION: 97.2%, TIME_FREQ: 96.8%, PLATFORM: 100%."
- **Empirical Ground Truth (`models/ner/test_metrics.json`):**
  - **LOCATION:** Support = 13, True Positives = 12, False Positives = 0, False Negatives = 1.
    $$\text{Precision} = 1.0000 \quad (100.0\%), \quad \text{Recall} = \frac{12}{13} \approx 0.9231 \quad (92.31\%), \quad \text{F1} = 0.9600 \quad (96.00\%)$$
  - **TIME_FREQ:** Support = 17, True Positives = 17, False Positives = 0, False Negatives = 0.
    $$\text{Precision} = 1.0000 \quad (100.0\%), \quad \text{Recall} = \frac{17}{17} = 1.0000 \quad (100.0\%), \quad \text{F1} = 1.0000 \quad (100.0\%)$$
- **Impact Analysis:**
  Reporting 97.2% and 96.8% in viva or a research paper represents unverified synthetic estimates. The true metrics are actually *better* for `TIME_FREQ` (100% vs 96.8%) and slightly lower for `LOCATION` (92.31% vs 97.2%) due to a single false negative span in 13 samples.
- **Recommended Exact Text Replacement:**
  ```markdown
  ### Empirical NER Test Set Evaluation Results (60 Held-Out Samples)
  - **Overall Micro-Averaged:** Precision: 100.0% (65/65), Recall: 98.48% (65/66), F1-Score: 99.24%
  - **PERP_REL (24 instances):** Precision: 100.0%, Recall: 100.0% (24/24), F1-Score: 100.0%
  - **LOCATION (13 instances):** Precision: 100.0%, Recall: 92.31% (12/13, 1 FN), F1-Score: 96.00%
  - **TIME_FREQ (17 instances):** Precision: 100.0%, Recall: 100.0% (17/17), F1-Score: 100.0%
  - **PLATFORM (12 instances):** Precision: 100.0%, Recall: 100.0% (12/12), F1-Score: 100.0%
  ```

---

### CORR-002: Production Database Seeding Scripts
- **Target Files:**
  - `docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md` (Section 3.3, Lines 115–119)
- **Severity:** **CRITICAL**
- **Documented Statement:**
  ```powershell
  # 3. Seed Verified Statutory Laws (32 Rows)
  python scripts/import_laws.py

  # 4. Seed Verified Support Helplines (12 Centers)
  python scripts/import_support_services.py
  ```
- **Empirical Ground Truth:**
  The directory `scripts/` does not contain `import_laws.py` or `import_support_services.py`. All statutory data validation, schema verification, and database persistence are centralized in `backend/app/services/data_importer.py` through `DatasetImporter`.
- **Impact Analysis:**
  A developer following the documented deployment playbook will encounter `FileNotFoundError` during database seeding on Render or local setup.
- **Recommended Exact Text Replacement:**
  ```powershell
  # 3. Seed Verified Statutory Laws (32 Rows) & Support Helplines (12 Centers)
  python -m app.services.data_importer
  ```
  *(Or execute via Python console: `from app.services.data_importer import DatasetImporter; DatasetImporter().run_all()`)*

---

### CORR-003: Database Relational Schema & ORM Definition
- **Target Files:**
  - `docs/development_guide/07_BACKEND_AND_DATABASE_DEVELOPMENT.md` (Section 2, DDL and Schemas)
- **Severity:** **MODERATE**
- **Documented Statement:**
  Documented theoretical DDL with UUID primary keys (`id UUID PRIMARY KEY`), PostgreSQL array types (`VARCHAR(50)[]`), and legal categorization fields (`cognizable BOOLEAN`, `bailable BOOLEAN`, `compoundable BOOLEAN`, `punishment_summary TEXT` in `laws`; `city`, `address`, `pin_code`, `operational_hours` in `support_services`; `email`, `hashed_password` in `users`; `pdf_download_url` in `reports`).
- **Empirical Ground Truth (`backend/app/models/`):**
  SQLAlchemy ORM models use Integer autoincrement primary keys (`id = Column(Integer, primary_key=True)`), pragmatic text columns, and JSON columns for extensible session states:
  - `Law` (`laws`): `id` (Integer), `act_name` (String), `section_number` (String), `section_text` (Text), `applicable_label` (String), `created_at` (DateTime).
  - `SupportService` (`support_services`): `id` (Integer), `service_type` (String), `name` (String), `contact_number` (String), `state` (String), `district` (String), `applicable_label` (String), `created_at` (DateTime).
  - `User` (`users`): `id` (Integer), `user_guid` (String), `is_anonymous` (Boolean), `created_at` (DateTime). (Zero survivor email or password stored).
  - `Report` (`reports`): `id` (Integer), `submission_id` (String), `report_data` (JSON), `created_at`, `updated_at`.
- **Impact Analysis:**
  A developer trying to run the documented DDL directly would build a schema incompatible with `DatasetImporter` and `Law` / `SupportService` models.
- **Recommended Exact Text Replacement:**
  Document the actual implemented SQLAlchemy models as the active primary schema, noting that the expanded legal classification columns (`bailable`, `cognizable`, `compoundable`) represent a planned future extension.

---

### CORR-004: Dataset Split File Locations
- **Target Files:**
  - `docs/development_guide/02_ENVIRONMENT_AND_PROJECT_SETUP.md` (Project Directory Tree)
  - `docs/development_guide/03_DATASET_AND_DATA_PREPARATION.md`
- **Severity:** **MODERATE**
- **Documented Statement:**
  Directory tree listed `data/train.csv`, `data/val.csv`, and `data/test.csv`.
- **Empirical Ground Truth:**
  The `data/` directory contains only the original root source files (`womens_safety_dataset.csv`, `laws.csv`, `support_services.csv`, `entity_extraction_dataset.json`). The stratified train, validation, and test splits generated by `ml/preprocessing/prepare_classification.py` are saved in:
  - `ml/preprocessing/train_data.csv` (760 records)
  - `ml/preprocessing/val_data.csv` (162 records)
  - `ml/preprocessing/test_data.csv` (164 records)
- **Impact Analysis:**
  Developers attempting to load training data from `data/train.csv` will receive file missing errors.
- **Recommended Exact Text Replacement:**
  Update directory diagrams to reflect:
  ```text
  ml/
  ├── preprocessing/
  │   ├── prepare_classification.py
  │   ├── train_data.csv        # 760 rows (70%)
  │   ├── val_data.csv          # 162 rows (15%)
  │   ├── test_data.csv         # 164 rows (15%)
  │   └── label_mapping.json
  ```

---

### CORR-005: Model Directory Naming & Fast Tokenizer Artifacts
- **Target Files:**
  - `docs/development_guide/02_ENVIRONMENT_AND_PROJECT_SETUP.md`
  - `docs/development_guide/05_BERT_MODEL_DEVELOPMENT.md`
- **Severity:** **MODERATE**
- **Documented Statement:**
  Listed `models/bert_multilabel_v1/` and listed `vocab.txt` inside `models/bert_multilabel_experiment_v2/`.
- **Empirical Ground Truth:**
  1. The legacy v1 model directory on disk is named `models/bert_multilabel/` (without `_v1`).
  2. The production v2 model directory is `models/bert_multilabel_experiment_v2/` and contains `model.safetensors` (417.68 MB), `config.json`, `label_mapping.json`, `optimal_thresholds.json`, `tokenizer.json`, `tokenizer_config.json`, and `training_config.json`. Standalone `vocab.txt` does not exist because HuggingFace `AutoTokenizer` uses `tokenizer.json`.
- **Recommended Exact Text Replacement:**
  Rename references from `models/bert_multilabel_v1/` to `models/bert_multilabel/`, and remove `vocab.txt` from the v2 artifact listing.

---

### CORR-006: Frontend Application Pages & Component Structure
- **Target Files:**
  - `docs/development_guide/02_ENVIRONMENT_AND_PROJECT_SETUP.md`
  - `docs/development_guide/10_FRONTEND_AND_API_INTEGRATION.md`
- **Severity:** **MODERATE**
- **Documented Statement:**
  Listed theoretical pages `IncidentReport.jsx`, `QuestionInterface.jsx`, `LegalInfo.jsx`, `SupportFinder.jsx` and components `IncidentForm.jsx`, `QuestionCard.jsx`, `EntityTag.jsx`, `LegalCard.jsx`, `SupportCard.jsx`.
- **Empirical Ground Truth (`frontend/src/`):**
  The actual frontend implements a 4-route SPA architecture in `frontend/src/App.jsx`:
  - `frontend/src/pages/Home.jsx` (`/`)
  - `frontend/src/pages/Chatbot.jsx` (`/chatbot`)
  - `frontend/src/pages/Processing.jsx` (`/processing`)
  - `frontend/src/pages/Report.jsx` (`/report`)
  Components in `frontend/src/components/` are:
  `AbheraEmblem.jsx`, `ChatInput.jsx`, `ChatMessage.jsx`, `HeroIllustration.jsx`, `LoadingIndicator.jsx`, `Navbar.jsx`.
  Legal cards, helpline accordions, and PDF generation are cleanly embedded directly within `Report.jsx` using `lucide-react` icons and `jspdf`.
- **Recommended Exact Text Replacement:**
  Update component tree and route diagrams in Modules 02 and 10 to reflect the 4 production pages and 6 modular UI components.

---

### CORR-007: Frontend & Backend Environment Variable Naming
- **Target Files:**
  - `docs/development_guide/02_ENVIRONMENT_AND_PROJECT_SETUP.md`
  - `docs/development_guide/10_FRONTEND_AND_API_INTEGRATION.md`
  - `docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md`
- **Severity:** **MODERATE**
- **Documented Statement:**
  - Frontend: `VITE_API_BASE_URL`
  - Backend: `ALLOWED_ORIGINS`
- **Empirical Ground Truth:**
  - In `frontend/src/services/api.js`: The code reads `import.meta.env.VITE_API_URL`.
  - In `backend/app/config.py`: The Pydantic setting attribute is `CORS_ORIGINS`.
- **Recommended Exact Text Replacement:**
  Update all `.env` templates and deployment instructions:
  - Frontend: `VITE_API_URL=https://abhera-api.onrender.com`
  - Backend: `CORS_ORIGINS=["https://abhera.vercel.app"]`

---

### CORR-008: Mermaid Mindmap Renderer Compatibility
- **Target Files:**
  - `docs/development_guide/09_LEGAL_SUPPORT_AND_ANTI_HALLUCINATION.md` (Line 32)
- **Severity:** **MINOR**
- **Documented Statement:**
  Uses experimental Mermaid `mindmap` diagram syntax for the statutory knowledge base.
- **Empirical Ground Truth:**
  Standard Markdown parsers, GitHub GFM, and IDE previewers do not reliably support `mindmap`, causing blank rendering.
- **Recommended Exact Replacement:**
  Convert the `mindmap` block into a standard, fully supported `flowchart TD` block:
  ```mermaid
  flowchart TD
      KB["ABHERA Statutory Knowledge Base"]
      KB --> BNS["Bharatiya Nyaya Sanhita, 2023 (BNS)"]
      KB --> IT["Information Technology Act, 2000 (IT Act)"]
      KB --> DV["Domestic Violence Act, 2005 (PWDVA)"]
      KB --> POSH["POSH Act, 2013"]
      KB --> DOWRY["Dowry Prohibition Act, 1961"]

      BNS --> BNS_69["Sec 69: False Promise of Marriage"]
      BNS --> BNS_74["Sec 74: Outraging Modesty (354 IPC)"]
      BNS --> BNS_75["Sec 75: Sexual Harassment (354A IPC)"]
      BNS --> BNS_76["Sec 76: Disrobing Assault (354B IPC)"]
      BNS --> BNS_77["Sec 77: Voyeurism (354C IPC)"]
      BNS --> BNS_78["Sec 78: Stalking (354D IPC)"]
      BNS --> BNS_79["Sec 79: Insulting Modesty / Obscene Words (509 IPC)"]
      BNS --> BNS_85["Sec 85 & 86: Marital Cruelty & In-Law Abuse (498A IPC)"]

      IT --> IT_66C["Sec 66C: Identity Theft / Impersonation"]
      IT --> IT_66E["Sec 66E: Violation of Bodily Privacy"]
      IT --> IT_67["Sec 67 & 67A: Transmitting Obscene / Sexually Explicit Acts"]

      DV --> DV_18["Sec 18: Protection Orders"]
      DV --> DV_19["Sec 19: Residence Orders"]
      DV --> DV_20["Sec 20: Monetary Relief"]

      POSH --> POSH_4["Sec 4: Internal Complaints Committee (ICC)"]
      POSH --> POSH_9["Sec 9: Inquiry Procedure & Employment Relief"]

      DOWRY --> DP_3["Sec 3 & 4: Giving, Taking & Demanding Dowry"]
  ```

---

### CORR-009: Unquoted Parentheses in Mermaid Database Cylinders
- **Target Files:**
  - `docs/development_guide/01_PROJECT_ORIGIN_AND_REQUIREMENTS.md` (Lines 77, 78)
  - `docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md` (Line 24)
- **Severity:** **MINOR**
- **Documented Statement:**
  `PostgresDB[(PostgreSQL 16 High-Availability Instance<br/>- Encrypted Storage at Rest (AES-256))]`
- **Empirical Ground Truth:**
  Parentheses like `(AES-256)` inside cylindrical brackets `[(...)]` cause syntax confusion in some Mermaid parser versions.
- **Recommended Exact Text Replacement:**
  Enclose inner text in double quotes:
  `PostgresDB[("PostgreSQL 16 High-Availability Instance<br/>- Encrypted Storage at Rest (AES-256)")]`

---

### CORR-010: Test Suite Virtual Environment Execution Command
- **Target Files:**
  - `docs/development_guide/12_TESTING_EVALUATION_AND_MODEL_PROMOTION.md`
  - `ABHERA_COMPLETE_DEVELOPMENT_AND_WORKING_GUIDE.md`
- **Severity:** **MINOR**
- **Documented Statement:**
  Instructed running `python -m unittest discover tests`.
- **Empirical Ground Truth:**
  Running with system global Python fails with `ModuleNotFoundError: No module named 'pydantic_settings'`. The project-specific virtual environment Python must be explicitly targeted.
- **Recommended Exact Text Replacement:**
  ```powershell
  # Windows PowerShell
  & "D:\Abhera(Mini)\backend\venv\Scripts\python.exe" -m unittest discover tests
  ```

---

### CORR-011: Model Inference Class Names
- **Target Files:**
  - `docs/development_guide/05_BERT_MODEL_DEVELOPMENT.md`
  - `docs/development_guide/10_FRONTEND_AND_API_INTEGRATION.md`
- **Severity:** **MINOR**
- **Documented Statement:**
  Referred to `HarassmentClassifier`.
- **Empirical Ground Truth:**
  In `ml/inference/predict_bert.py`, the class is `BERTIncidentPredictor`.
  In `ml/inference/predict_ner.py`, the class is `NEREntityPredictor`.
- **Recommended Exact Text Replacement:**
  Use `BERTIncidentPredictor` and `NEREntityPredictor`.

---

### CORR-012: Database Name Configuration
- **Target Files:**
  - `docs/development_guide/07_BACKEND_AND_DATABASE_DEVELOPMENT.md`
  - `docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md`
- **Severity:** **MINOR**
- **Documented Statement:**
  Mentioned `abhera_db` as the local default database.
- **Empirical Ground Truth:**
  In `backend/app/config.py`, the default setting is `POSTGRES_DB: str = os.getenv("POSTGRES_DB", "womens_safety_db")`.
- **Recommended Exact Text Replacement:**
  Note that local development defaults to `womens_safety_db`, while hosted Render deployments can use any database name provided in the `DATABASE_URL` connection string.

---

## 3. Verification & Compliance Sign-Off

This correction log provides the complete, authoritative, and empirical reconciliation between the documentation suite and the physical ABHERA codebase at `D:\Abhera(Mini)`.
All findings are backed by automated tests, file scans, and evaluation JSON artifacts.
