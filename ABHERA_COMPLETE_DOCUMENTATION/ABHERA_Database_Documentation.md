# ABHERA Relational Database Specification & Data Dictionary
**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**RDBMS Platform:** PostgreSQL 16 (Relational ACID Compliant)  
**ORM Framework:** SQLAlchemy 2.0 (Declarative Base)  
**Default Database Name:** `womens_safety_db` (Configurable via `DATABASE_URL`)  
**Active Implemented Tables:** 11 Normalized Tables  

---

## 1. Database Architectural Principles & Schema Topology

The ABHERA database architecture is designed to support high-throughput conversational state persistence, immutable audit logging of neural model predictions, and deterministic retrieval of vetted statutory provisions.

```
+=======================================================================================================+
|                            POSTGRESQL 11-TABLE RELATIONAL ENTITY TOPOLOGY                             |
+=======================================================================================================+

        +-------------------+
        |       USERS       |
        |  id (PK, Integer) |
        |  user_guid        |
        |  is_anonymous     |
        +-------------------+
                  | 1
                  |
                  | owns
                  v *
        +-------------------------------------------------------------------+
        |                             SESSIONS                              |
        |  id (PK, Integer)           session_id (UK, String)               |
        |  user_id (FK -> users)      submission_id (String)                |
        |  messages (JSON)            answers (JSON)                        |
        |  predicted_labels (JSON)    entities (JSON)                       |
        |  current_question (JSON)    conversation_status (String)          |
        +-------------------------------------------------------------------+
                  | 1
                  |
                  | originates
                  v 1
        +-------------------------------------------------------------------+
        |                       INCIDENT_SUBMISSIONS                        |
        |  id (PK, Integer)           submission_id (UK, String)            |
        |  session_id (FK -> sessions)narrative_text (Text)                 |
        |  predicted_labels (JSON)    report_generated (Boolean)            |
        +-------------------------------------------------------------------+
           |                     |                      |                |
           | 1                   | 1                    | 1              | 1
           | logs                | generates            | contains       | compiles
           v *                   v *                    v *              v 1
    +---------------+    +---------------+      +---------------+   +----------------+
    | CONVERSATION_ |    |  PREDICTIONS  |      |   ENTITIES    |   |    REPORTS     |
    |   MESSAGES    |    | id (PK)       |      | id (PK)       |   | id (PK)        |
    | id (PK)       |    | submission_id |      | submission_id |   | submission_id  |
    | submission_id |    | label (DV,SH) |      | entity_type   |   | report_data    |
    | sender        |    | conf_score    |      | entity_value  |   | (10-Sec JSON)  |
    | message_text  |    | is_positive   |      | start/end_char|   +----------------+
    +---------------+    +---------------+      +---------------+
                                                        |
    +---------------------------------------------------+
    |
    | prompts / records answers
    v
  +-------------------------------------+       +------------------------------------+
  |              QUESTIONS              |       |    SUBMISSION_QUESTION_ANSWERS     |
  |  question_id (PK, String(64))       | <===> |  id (PK, Integer)                  |
  |  required_entity (LOCATION, etc.)   |       |  submission_id (FK -> submissions) |
  |  category (General, DV, Cyber)      |       |  question_id (FK -> questions)     |
  |  question_text (Text)               |       |  answer (Text)                     |
  |  priority (Integer) | active (Bool) |       |  asked_at / answered_at (DateTime) |
  +-------------------------------------+       +------------------------------------+

  ============================== CANONICAL KNOWLEDGE BASE ===============================

        +-----------------------------------+   +------------------------------------+
        |               LAWS                |   |          SUPPORT_SERVICES          |
        |  id (PK, Integer)                 |   |  id (PK, Integer)                  |
        |  act_name (BNS, IT, POSH, DV)     |   |  service_type (Emergency, Helpline)|
        |  section_number (Sec 69, 78, etc.)|   |  name (Sakhi OSC, NCW Helpline)    |
        |  section_text (Verbatim Statute)  |   |  contact_number (112, 181, 1930)   |
        |  applicable_label (DV, SH, ST, CA)|   |  state / district (Geospatial)     |
        |  created_at (DateTime)            |   |  applicable_label (Category Filter)|
        |  -- 32 Verified Statutory Rows -- |   |  -- 12 Verified Facility Records --|
        +-----------------------------------+   +------------------------------------+
```

---

## 2. Table-by-Table Data Dictionary

### 2.1 Table: `users`
- **Model File:** `backend/app/models/user.py`
- **Purpose:** Strict privacy-preserving identity management. To protect survivor safety and eliminate legal subpoena risks, ABHERA stores zero survivor emails, passwords, or personal credentials.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement): Internal sequence identifier.
  - `user_guid` (String(128), Unique, Indexed, Default `uuid4`): Pseudonymous surrogate identifier.
  - `is_anonymous` (Boolean, Default True): Explicit assertion of anonymous survivor status.
  - `created_at` (DateTime, Default UTC now): Timestamp of account initiation.

---

### 2.2 Table: `sessions`
- **Model File:** `backend/app/models/session.py`
- **Purpose:** Multi-turn conversational state machine ledger tracking question queues and JSON context.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement): Relational ID.
  - `session_id` (String(128), Unique, Indexed): Cryptographic public UUID token shared with frontend.
  - `submission_id` (String(128), Indexed, Nullable): Linked incident submission reference.
  - `user_id` (Integer, ForeignKey `users.id`, Nullable): Optional user reference.
  - `messages` (JSON, Default `[]`): Array of turn dictionaries `[{"sender": "user", "text": "..."}]`.
  - `answers` (JSON, Default `{}`): Key-value store mapping forensic slots to survivor responses.
  - `predicted_labels` (JSON, Default `[]`): Array of active classification codes (`["DV", "CA"]`).
  - `entities` (JSON, Default `{}`): Grouped dictionary of extracted NER entities.
  - `current_question` (JSON, Nullable): Active question prompt awaiting user input.
  - `conversation_status` (String(64), Default `'ACTIVE'`): State transition flag (`ACTIVE`, `QUESTIONING`, `COMPLETED`).
  - `created_at` / `updated_at` (DateTime, Default UTC now): State audit timestamps.

---

### 2.3 Table: `incident_submissions`
- **Model File:** `backend/app/models/incident_submission.py`
- **Purpose:** Core intake record holding the cumulative survivor narrative and dossier compilation flag.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement): Internal sequence key.
  - `submission_id` (String(128), Unique, Indexed): Public reference identifier (e.g., `SUB_20261102_E4F1`).
  - `session_id` (Integer, ForeignKey `sessions.id`, Nullable): Parent session linkage.
  - `narrative_text` (Text, Nullable): Complete cumulative disclosure text.
  - `predicted_labels` (JSON, Nullable): Array of predicted harassment classes.
  - `report_generated` (Boolean, Default False): Flag indicating report compilation.
  - `created_at` / `updated_at` (DateTime, Default UTC now).

---

### 2.4 Table: `conversation_messages`
- **Model File:** `backend/app/models/conversation_message.py`
- **Purpose:** Immutable audit trail of every conversational turn.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `submission_id` (String(64), Indexed): Associated submission reference.
  - `sender` (String(32)): Either `'user'` or `'assistant'`.
  - `message_text` (Text): Literal message string.
  - `timestamp` (DateTime, Default UTC now).

---

### 2.5 Table: `predictions`
- **Model File:** `backend/app/models/prediction.py`
- **Purpose:** Machine learning audit log persisting raw probabilities and threshold comparisons.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `submission_id` (String(128), ForeignKey `incident_submissions.submission_id`, Indexed).
  - `label` (String(32)): Harassment code (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
  - `confidence_score` (Float): Computed Sigmoid probability $\hat{p}_c \in [0.0, 1.0]$.
  - `is_positive` (Boolean): True if confidence satisfies optimal threshold $	au_c$.
  - `timestamp` (DateTime, Default UTC now).

---

### 2.6 Table: `entities`
- **Model File:** `backend/app/models/entity.py`
- **Purpose:** Persists evidentiary token spans extracted by the fine-tuned BERT NER model.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `submission_id` (String(128), ForeignKey `incident_submissions.submission_id`, Indexed).
  - `entity_type` (String(64)): Category label (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`).
  - `entity_value` (String(255)): Literal character span text (e.g., `'husband'`, `'metro station'`).
  - `start_char` / `end_char` (Integer): Zero-indexed narrative character bounds.
  - `timestamp` (DateTime, Default UTC now).

---

### 2.7 Table: `questions`
- **Model File:** `backend/app/models/question.py`
- **Purpose:** Canonical question catalog seeded by `backend/app/db/seed_questions.py`.
- **Columns:**
  - `question_id` (String(64), Primary Key): Descriptive key (e.g., `'GEN_PERP_REL'`).
  - `required_entity` (String(64)): Target forensic slot filled by this question.
  - `category` (String(64)): Domain context (`'General'`, `'DV'`, `'Cyber'`).
  - `question_text` (Text): Trauma-informed prompt string.
  - `priority` (Integer, Default 1): Queue order ranking ($1 = 	ext{highest}$).
  - `active` (Boolean, Default True): Enablement flag.

---

### 2.8 Table: `submission_question_answers`
- **Model File:** `backend/app/models/question.py` (`SubmissionQuestionAnswer`)
- **Purpose:** Records user replies to dynamic follow-up questions.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `submission_id` (String(64), ForeignKey `incident_submissions.submission_id`, Indexed).
  - `question_id` (String(64), ForeignKey `questions.question_id`).
  - `answer` (Text): Survivor's literal answer text.
  - `asked_at` / `answered_at` (DateTime, Default UTC now).

---

### 2.9 Table: `laws`
- **Model File:** `backend/app/models/law.py`
- **Purpose:** Vetted repository of Indian criminal and civil statutes (populated from `data/laws.csv`).
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `act_name` (String(255), Indexed): Governing statute (e.g., `'Bharatiya Nyaya Sanhita, 2023'`).
  - `section_number` (String(64), Indexed): Statutory provision (e.g., `'Section 69'`, `'Section 78'`).
  - `section_text` (Text): Verbatim statutory definition of the legal offence.
  - `applicable_label` (String(32), Indexed): Mapped harassment category code (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
  - `created_at` (DateTime, Default UTC now).
- **Verified Record Count:** Exactly **32 statutory provisions**.

---

### 2.10 Table: `support_services`
- **Model File:** `backend/app/models/support_service.py`
- **Purpose:** Institutional support helplines and One-Stop Crisis Centers (populated from `data/support_services.csv`).
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `service_type` (String(64), Indexed): Service category (`'Emergency'`, `'Helpline'`, `'One Stop Centre'`).
  - `name` (String(255), Indexed): Facility designation (e.g., `'Sakhi One Stop Centre'`).
  - `contact_number` (String(64)): Verified telephone dial string (`112`, `181`, `1091`, `1930`).
  - `state` (String(64), Indexed): State jurisdiction (or `'National'`).
  - `district` (String(64), Indexed): District jurisdiction (or `'All'`).
  - `applicable_label` (String(32), Indexed): Primary category association.
  - `created_at` (DateTime, Default UTC now).
- **Verified Record Count:** Exactly **12 facility records**.

---

### 2.11 Table: `reports`
- **Model File:** `backend/app/models/report.py`
- **Purpose:** Stores the immutable 10-section JSON payload representing the final compiled legal and safety report.
- **Columns:**
  - `id` (Integer, Primary Key, Autoincrement).
  - `submission_id` (String(128), Unique, Indexed): Associated submission reference.
  - `report_data` (JSON): Complete 10-section structured JSON dossier dictionary.
  - `created_at` / `updated_at` (DateTime, Default UTC now).

---

## 3. Planned Future Relational Extensions (Non-Breaking Roadmap)

In planned institutional releases, the following non-breaking enhancements will be migrated:
1. **`laws` Table Extensions:** Addition of structured procedural columns:
   - `bailable BOOLEAN DEFAULT FALSE`
   - `cognizable BOOLEAN DEFAULT TRUE`
   - `compoundable BOOLEAN DEFAULT FALSE`
   - `court_type VARCHAR(100)` (e.g., `'Court of Session'`)
   - `punishment_summary TEXT`
   *(Currently, these procedural elements are embedded directly within `section_text` and verified through deterministic string assertions).*
2. **`support_services` Table Extensions:** Addition of operational attributes:
   - `operating_hours VARCHAR(100) DEFAULT '24x7'`
   - `pin_code VARCHAR(10)`
   - `gps_coordinates POINT`
3. **`reports` Table Extensions:** Cloud storage signed URL pointer:
   - `pdf_download_url VARCHAR(512)`
