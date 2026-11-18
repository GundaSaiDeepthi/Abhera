"""
Module for Chapter 7: Database Design
"""

def get_chapter_07():
    return r'''# CHAPTER 7: Database Design

---

## 7.1 What is PostgreSQL and Why is it Used?

**PostgreSQL** is an advanced, enterprise-grade, open-source object-relational database management system (ORDBMS).

In a safety-critical application like ABHERA, data integrity and deterministic retrieval are non-negotiable. We cannot use purely NoSQL document stores (like MongoDB) because they lack strict relational schemas and foreign-key cascade constraints. We cannot use SQLite in production because it locks the entire database file during writes, creating bottlenecks under concurrent user traffic.

PostgreSQL was chosen for four primary architectural reasons:
1. **Strict ACID Compliance**: Atomicity, Consistency, Isolation, and Durability guarantee that transactions are never partially completed or corrupted.
2. **Referential Integrity**: Foreign-key relationships guarantee that orphan messages, predictions, or reports cannot exist without a valid parent submission.
3. **Advanced JSON Types**: PostgreSQL supports native `JSON` columns, allowing ABHERA to store flexible conversational message lists, extracted entity dictionaries, and multi-section report objects while maintaining strict relational boundaries.
4. **B-Tree Indexing**: Indexed columns (`applicable_label`, `session_id`, `submission_id`) ensure that statutory legal provisions are retrieved in less than 2 milliseconds.

---

## 7.2 Database Creation and Setup

The database is created in PostgreSQL with UTF-8 character encoding to support all characters safely:

```sql
-- PostgreSQL Database Creation Command
CREATE DATABASE womens_safety_db
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'English_United States.1252'
    LC_CTYPE = 'English_United States.1252'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
```

In the codebase, schema initialization is performed programmatically using SQLAlchemy metadata in `backend/app/db/init_db.py`:
```python
from app.database import engine, Base
import app.models  # Imports all 11 models

def init_db():
    Base.metadata.create_all(bind=engine)
```

---

## 7.3 Detailed Table Schema Analysis (All 11 Tables)

ABHERA contains exactly **11 relational tables** in PostgreSQL:

### Table 1: `users`
- **Purpose**: Tracks anonymous user identities without collecting personally identifiable information (PII).
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `user_guid`: `VARCHAR(64)`, Unique, Indexed, Default UUID.
  - `is_anonymous`: `BOOLEAN`, Default `TRUE`, Not Null.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Server Default `now()`.
- **Relationships**: One user has many sessions (`sessions.user_id`).

### Table 2: `sessions`
- **Purpose**: Stores active multi-turn conversational session states.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `session_id`: `VARCHAR(64)`, Unique, Indexed, Format: `SESS-<uuid[:8]>`.
  - `submission_id`: `VARCHAR(64)`, Nullable, Indexed.
  - `user_id`: `INTEGER`, Foreign Key referencing `users.id` (`ON DELETE SET NULL`).
  - `messages`: `JSON`, Default `[]`. Stores message history list.
  - `answers`: `JSON`, Default `[]`. Stores answered questions.
  - `predicted_labels`: `JSON`, Default `[]`. Stores model classifications.
  - `entities`: `JSON`, Default `{}`. Stores extracted NER entities.
  - `current_question`: `JSON`, Nullable. Stores active dynamic question.
  - `conversation_status`: `VARCHAR(30)`, Default `'ACTIVE'`. Values: `ACTIVE`, `QUESTIONING`, `PROCESSING`, `COMPLETED`.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.
  - `updated_at`: `TIMESTAMP WITH TIME ZONE`, Auto-updated on modification.

### Table 3: `incident_submissions`
- **Purpose**: Central record for a reported harassment incident.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Unique, Indexed, Format: `SUB-<uuid[:8]>`.
  - `session_id`: `VARCHAR(64)`, Foreign Key referencing `sessions.session_id` (`ON DELETE CASCADE`).
  - `narrative_text`: `TEXT`, Not Null. The raw survivor statement.
  - `state`: `VARCHAR(100)`, Nullable. Geospatial state (e.g., `'Telangana'`).
  - `district`: `VARCHAR(100)`, Nullable. Geospatial district (e.g., `'Hyderabad'`).
  - `predicted_labels`: `JSON`, Nullable.
  - `extracted_entities`: `JSON`, Nullable.
  - `report_generated`: `BOOLEAN`, Default `FALSE`.
  - `timestamp`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 4: `conversation_messages`
- **Purpose**: Granular relational audit log of every individual message turn.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Foreign Key referencing `incident_submissions.submission_id` (`ON DELETE CASCADE`).
  - `sender`: `VARCHAR(20)`, Not Null. Values: `'USER'`, `'SYSTEM'`, `'ASSISTANT'`.
  - `message_text`: `TEXT`, Not Null.
  - `timestamp`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 5: `predictions`
- **Purpose**: Stores individual model prediction confidence scores for auditing.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Foreign Key referencing `incident_submissions.submission_id` (`ON DELETE CASCADE`).
  - `model_name`: `VARCHAR(100)`, Default `'bert-base-uncased'`.
  - `category`: `VARCHAR(50)`, Not Null (e.g., `'ST'`, `'CA'`).
  - `probability`: `FLOAT`, Not Null (e.g., `0.8845`).
  - `is_above_threshold`: `BOOLEAN`, Not Null.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 6: `entities`
- **Purpose**: Stores individual extracted NER entity spans relationally.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Foreign Key referencing `incident_submissions.submission_id` (`ON DELETE CASCADE`).
  - `entity_type`: `VARCHAR(50)`, Not Null (e.g., `'PERP_REL'`, `'PLATFORM'`).
  - `entity_text`: `VARCHAR(255)`, Not Null (e.g., `'manager'`, `'Instagram'`).
  - `confidence_score`: `FLOAT`, Nullable.
  - `start_char`: `INTEGER`, Nullable.
  - `end_char`: `INTEGER`, Nullable.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 7: `questions`
- **Purpose**: Master bank of predefined forensic follow-up inquiries.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `question_id`: `VARCHAR(50)`, Unique, Indexed (e.g., `'Q_SAFETY_01'`).
  - `category`: `VARCHAR(50)`, Not Null (e.g., `'Stalking'`, `'General'`).
  - `question_text`: `TEXT`, Not Null.
  - `required_entity`: `VARCHAR(50)`, Not Null (e.g., `'TIME_FREQ'`, `'LOCATION'`).
  - `priority`: `INTEGER`, Default `100`. Higher priority asked first.
  - `active`: `BOOLEAN`, Default `TRUE`.

### Table 8: `submission_question_answers`
- **Purpose**: Links answered questions to incident submissions.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Foreign Key referencing `incident_submissions.submission_id` (`ON DELETE CASCADE`).
  - `question_id`: `VARCHAR(50)`, Foreign Key referencing `questions.question_id`.
  - `answer`: `TEXT`, Not Null. The survivor's answer.
  - `answered_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 9: `laws` (Knowledge Base)
- **Purpose**: Master repository of verified Indian statutory provisions.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `act_name`: `VARCHAR(100)`, Indexed (e.g., `'BNS'`, `'IT_ACT'`, `'POSH_ACT'`).
  - `section_number`: `VARCHAR(50)`, Not Null (e.g., `'Section 78'`, `'Section 66E'`).
  - `section_text`: `TEXT`, Not Null. Official statutory legal wording.
  - `applicable_label`: `VARCHAR(50)`, Indexed (e.g., `'ST'`, `'CA'`, `'DV'`, `'SH'`).
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 10: `support_services` (Knowledge Base)
- **Purpose**: Master directory of verified emergency hotlines and support units.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `service_type`: `VARCHAR(50)`, Indexed (e.g., `'EMERGENCY'`, `'CYBERCRIME'`, `'LEGAL_AID'`).
  - `name`: `VARCHAR(255)`, Not Null (e.g., `'National Emergency Number (112)'`).
  - `contact_number`: `VARCHAR(255)`, Not Null (e.g., `'112'`, `'181'`).
  - `state`: `VARCHAR(100)`, Indexed (e.g., `'ALL'`, `'Telangana'`).
  - `district`: `VARCHAR(100)`, Indexed.
  - `applicable_label`: `VARCHAR(255)`, Indexed.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.

### Table 11: `reports`
- **Purpose**: Permanent storage of compiled, immutable incident report JSON dossiers.
- **Columns**:
  - `id`: `INTEGER`, Primary Key, Autoincrement.
  - `submission_id`: `VARCHAR(64)`, Unique, Indexed, Foreign Key referencing `incident_submissions.submission_id` (`ON DELETE CASCADE`).
  - `report_data`: `JSON`, Not Null. Complete 10-section structured JSON document.
  - `created_at`: `TIMESTAMP WITH TIME ZONE`, Default `now()`.
  - `updated_at`: `TIMESTAMP WITH TIME ZONE`, Auto-updated.

---

## 7.4 Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    users ||--o{ sessions : "creates"
    sessions ||--o| incident_submissions : "links_to"
    incident_submissions ||--o{ conversation_messages : "contains"
    incident_submissions ||--o{ predictions : "has"
    incident_submissions ||--o{ entities : "extracts"
    incident_submissions ||--o{ submission_question_answers : "answers"
    questions ||--o{ submission_question_answers : "referenced_by"
    incident_submissions ||--o| reports : "compiles_into"

    laws ||..o{ incident_submissions : "statutorily_mapped_to"
    support_services ||..o{ incident_submissions : "geospatially_routed_to"
```

---

## 7.5 SQL Examples from the Project

### 1. Inserting a New Anonymous Session:
```sql
INSERT INTO sessions (session_id, submission_id, conversation_status, created_at, updated_at)
VALUES ('SESS-3f8a12bc', 'SUB-9d7e41fa', 'ACTIVE', NOW(), NOW())
RETURNING id, session_id;
```

### 2. Querying Verified Laws for Predicted Incident Category ('ST' - Stalking):
```sql
SELECT act_name, section_number, section_text, applicable_label
FROM laws
WHERE applicable_label = 'ST'
ORDER BY id ASC;
```

### 3. Inserting Answer to Dynamic Follow-up Question:
```sql
INSERT INTO submission_question_answers (submission_id, question_id, answer, answered_at)
VALUES ('SUB-9d7e41fa', 'Q_TIME_FREQ_01', 'It has been happening every day for the past 3 weeks', NOW())
ON CONFLICT (submission_id, question_id) 
DO UPDATE SET answer = EXCLUDED.answer, answered_at = NOW();
```

### 4. Fetching Support Services for a Specific State and Category:
```sql
SELECT service_type, name, contact_number, state, district
FROM support_services
WHERE (state = 'Telangana' OR state = 'ALL')
  AND (applicable_label LIKE '%ST%' OR applicable_label LIKE '%ALL EMERGENCIES%')
ORDER BY id ASC;
```
'''
