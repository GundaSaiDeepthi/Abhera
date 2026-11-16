# ABHERA REST API Engineering Reference Manual
**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**Specification Standard:** OpenAPI 3.0 / RESTful HTTP JSON  
**Base URL (Local Development):** `http://localhost:8000`  
**Base URL (Production Container):** `https://abhera-api.onrender.com`  
**Authentication Scheme:** Pseudonymous Session GUID Tokens (`session_id`)  

---

## 1. API Architecture & Global Configurations

ABHERA exposes a high-performance, asynchronous REST API powered by **FastAPI** (`backend/app/main.py`). The API design adheres strictly to stateless HTTP semantics, standard HTTP response status codes, and comprehensive Pydantic payload serialization.

### 1.1 CORS Security Configuration
To protect cross-origin survivor disclosures while enabling decoupled frontend hosting on Vercel, CORS origins are governed by `backend/app/config.py`:
```python
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
```
During production deployment, the cloud domain (e.g., `https://abhera.vercel.app`) is dynamically appended via the `CORS_ORIGINS` environment variable.

### 1.2 Global HTTP Status Code Semantics
- `200 OK`: Request succeeded synchronously; structured payload returned.
- `400 Bad Request`: Payload validation failed (empty session ID, blank message, or invalid syntax).
- `404 Not Found`: Target resource does not exist (invalid `session_id` or non-existent `submission_id`).
- `422 Unprocessable Entity`: Request body failed strict Pydantic type validation schema.
- `500 Internal Server Error`: Unhandled server-side exception (database connection failure or model crash).

---

## 2. Exhaustive API Endpoint Catalog

| HTTP Method | Route Endpoint | Purpose / Description | Request Schema | Response Schema | Related Service Layer |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **GET** | `/` | Root service introduction & documentation links | None | JSON Dict | `backend/app/main.py` |
| **GET** | `/health` | Uptime health check & diagnostic verification | None | `HealthResponse` | `backend/app/routes/health.py` |
| **POST** | `/api/session/start` | Initiates an anonymous survivor session token | None | `SessionStartResponse` | `app/services/session_service.py` |
| **GET** | `/api/session/{session_id}` | Retrieves active conversation history & state | URL Param: `session_id` | `SessionDetailResponse` | `app/services/session_service.py` |
| **POST** | `/api/chat/message` | Core conversational intake & interview turn | `ChatMessageRequest` | `ChatMessageResponse` | `app/services/chat_service.py` |
| **GET** | `/api/report/{submission_id}` | Retrieves immutable 10-section report dossier | URL Param: `submission_id` | `ReportResponse` | `app/services/report_generator.py` |

---

## 3. Endpoint Specifications & Payload Schemas

### 3.1 Root Service Endpoint
- **Method:** `GET`
- **Path:** `/`
- **Description:** Returns welcome metadata and links to interactive Swagger UI documentation.
- **Example Response (`200 OK`):**
  ```json
  {
    "message": "Welcome to the Conversational Women's Safety Legal Assistance System API",
    "docs_url": "/docs",
    "health_check": "/health"
  }
  ```

---

### 3.2 Service Health Check
- **Method:** `GET`
- **Path:** `/health`
- **Description:** Verifies service responsiveness, database connection pool vitality, and model loading state.
- **Example Response (`200 OK`):**
  ```json
  {
    "status": "healthy",
    "service": "Women's Safety Legal Assistance API",
    "version": "0.1.0",
    "timestamp": "2026-11-02T20:30:00Z"
  }
  ```

---

### 3.3 Start Anonymous Session
- **Method:** `POST`
- **Path:** `/api/session/start`
- **Headers:** `Content-Type: application/json`
- **Request Body:** None required.
- **Service Logic:**
  1. Generates a cryptographically strong UUID4 `session_id`.
  2. Generates an associated `submission_id` (e.g., `SUB_20261102_E4F1`).
  3. Creates an anonymous `User` record (`is_anonymous=True`).
  4. Persists an initial `Session` record with `conversation_status='ACTIVE'`.
- **Example Response (`200 OK`):**
  ```json
  {
    "session_id": "8f3b2e10-74a9-4b61-9c12-3456789abcde",
    "submission_id": "SUB_20261102_E4F1",
    "conversation_status": "ACTIVE",
    "created_at": "2026-11-02T20:30:15Z"
  }
  ```

---

### 3.4 Retrieve Session State
- **Method:** `GET`
- **Path:** `/api/session/{session_id}`
- **Path Parameters:**
  - `session_id` (string, required): The UUID session token stored in browser `sessionStorage`.
- **Example Response (`200 OK`):**
  ```json
  {
    "session_id": "8f3b2e10-74a9-4b61-9c12-3456789abcde",
    "submission_id": "SUB_20261102_E4F1",
    "conversation_status": "QUESTIONING",
    "predicted_labels": ["DV", "CA"],
    "extracted_entities": {
      "PERP_REL": ["husband"],
      "PLATFORM": ["WhatsApp"]
    },
    "current_question": {
      "question_id": "GEN_LOCATION",
      "required_entity": "LOCATION",
      "question_text": "Where did these incidents take place (e.g., at home, workplace, public transit, or online)?"
    },
    "message_count": 4,
    "created_at": "2026-11-02T20:30:15Z",
    "updated_at": "2026-11-02T20:31:45Z"
  }
  ```

---

### 3.5 Process Conversational Chat Message (Primary Intake)
- **Method:** `POST`
- **Path:** `/api/chat/message`
- **Headers:** `Content-Type: application/json`
- **Request Schema (`ChatMessageRequest`):**
  ```json
  {
    "session_id": "8f3b2e10-74a9-4b61-9c12-3456789abcde",
    "message": "My husband has been beating me every day at home and sending death threats on WhatsApp."
  }
  ```
- **Execution Workflow:**
  1. Validates `session_id` and non-empty `message`.
  2. Updates `incident_submissions` narrative text.
  3. Executes `predict_bert.py`: yields predicted classes `DV` and `CA`.
  4. Executes `predict_ner.py`: extracts entities:
     - `husband` $	o$ `PERP_REL` (chars 3–10)
     - `every day` $	o$ `TIME_FREQ` (chars 26–35)
     - `home` $	o$ `LOCATION` (chars 39–43)
     - `WhatsApp` $	o$ `PLATFORM` (chars 74–82)
  5. Dynamic Question Engine evaluates remaining forensic slots. If slots are satisfied, status transitions to `COMPLETED`.
  6. Maps verified Section 85/86 BNS (Domestic Cruelty) and Section 66C/67 IT Act.
  7. Queries Sakhi One Stop Centre (OSC) and Women Helpline (181).
- **Example Response (`200 OK`):**
  ```json
  {
    "session_id": "8f3b2e10-74a9-4b61-9c12-3456789abcde",
    "submission_id": "SUB_20261102_E4F1",
    "bot_response": "I understand how difficult this is. We have identified Domestic Violence and Cyber Harassment indicators. Your safety is our absolute priority.",
    "predicted_labels": ["DV", "CA"],
    "extracted_entities": {
      "PERP_REL": ["husband"],
      "TIME_FREQ": ["every day"],
      "LOCATION": ["home"],
      "PLATFORM": ["WhatsApp"]
    },
    "dynamic_question": null,
    "conversation_status": "COMPLETED",
    "report_ready": true,
    "recommended_actions": [
      "Contact Emergency Helpline 112 immediately if in immediate physical danger.",
      "Preserve WhatsApp screenshots without altering timestamps.",
      "Access confidential medical care and safe shelter at Sakhi One Stop Centre."
    ]
  }
  ```

---

### 3.6 Retrieve Incident Report Dossier
- **Method:** `GET`
- **Path:** `/api/report/{submission_id}`
- **Path Parameters:**
  - `submission_id` (string, required): Unique submission reference (e.g., `SUB_20261102_E4F1`).
- **Response Schema (`ReportResponse`):** Returns the complete 10-section structured JSON payload persisted in the `reports` table.
- **Example Response Payload (`200 OK`):**
  ```json
  {
    "status": "success",
    "submission_id": "SUB_20261102_E4F1",
    "report": {
      "section_1_header_and_overview": {
        "report_id": "REP_SUB_20261102_E4F1",
        "generated_at": "2026-11-02T20:32:00Z",
        "incident_narrative": "My husband has been beating me every day at home and sending death threats on WhatsApp.",
        "case_summary": "Multi-category incident comprising Domestic Violence and Cyber Harassment."
      },
      "section_2_primary_classifications": [
        {
          "code": "DV",
          "name": "Domestic Violence",
          "confidence_score": 0.9459,
          "decision_threshold": 0.73,
          "is_positive": true
        },
        {
          "code": "CA",
          "name": "Cyber Abuse",
          "confidence_score": 0.9750,
          "decision_threshold": 0.45,
          "is_positive": true
        }
      ],
      "section_3_extracted_information": {
        "PERP_REL": ["husband"],
        "LOCATION": ["home"],
        "TIME_FREQ": ["every day"],
        "PLATFORM": ["WhatsApp"]
      },
      "section_4_questionnaire_responses": [],
      "section_5_applicable_legal_provisions": [
        {
          "act_name": "Bharatiya Nyaya Sanhita, 2023",
          "section_number": "Section 85 & 86",
          "section_text": "Husband or relative of husband of a woman subjecting her to cruelty...",
          "applicable_label": "DV"
        },
        {
          "act_name": "Information Technology Act, 2000",
          "section_number": "Section 66C",
          "section_text": "Punishment for identity theft and sending threatening communications...",
          "applicable_label": "CA"
        }
      ],
      "section_6_recommended_support_services": [
        {
          "service_type": "Emergency",
          "name": "National Emergency Response Support System",
          "contact_number": "112",
          "state": "National",
          "district": "All"
        },
        {
          "service_type": "One Stop Centre",
          "name": "Sakhi One Stop Centre",
          "contact_number": "181",
          "state": "Telangana",
          "district": "Hyderabad"
        }
      ],
      "section_7_action_recommendations": [
        "Seek emergency shelter assistance if physical safety is threatened.",
        "File a Formal Information Report (FIR) under BNS Sections 85 and 86."
      ],
      "section_8_evidence_preservation_guide": [
        "Export raw WhatsApp chat history as an unedited text archive.",
        "Obtain certified Medical Legal Case (MLC) report from a government hospital."
      ],
      "section_9_model_prediction_information": {
        "classifier": "bert-base-uncased (Multi-Label BCE pos_weight)",
        "ner": "bert-base-uncased-token-classification",
        "data_provenance": "Legal provisions retrieved directly from PostgreSQL laws table."
      },
      "section_10_legal_disclaimer": {
        "disclaimer_text": "ABHERA is an educational legal information and safety support tool. It does not provide formal legal representation."
      }
    }
  }
  ```
