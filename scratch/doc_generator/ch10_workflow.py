"""
Module for Chapter 10: Complete ABHERA Workflow
"""

def get_chapter_10():
    return r'''# CHAPTER 10: Complete ABHERA Workflow

---

## 10.1 The 13-Step End-to-End User Journey

Every user interaction in ABHERA follows a verified, 13-step pipeline from initial browser access to the final PDF download:

```
[ Step 1: User Opens ABHERA ]
             |
             v
[ Step 2: User Enters Incident Narrative ]
             |
             v
[ Step 3: Frontend Sends POST /api/chat/message ]
             |
             v
[ Step 4: Backend Validates Payload & Session ]
             |
             v
[ Step 5: BERT Multi-Label Classification Head ]
             |
             v
[ Step 6: BERT NER Token Classification Head ]
             |
             v
[ Step 7: Dynamic Question Engine Selects Next Slot ]
             |
             v
[ Step 8: User Submits Follow-Up Answer ]
             |
             v
[ Step 9: State & Answers Persisted in PostgreSQL ]
             |
             v
[ Step 10: Anti-Hallucination Legal Retrieval from laws Table ]
             |
             v
[ Step 11: Emergency Support Retrieval from support_services Table ]
             |
             v
[ Step 12: Final 10-Section Dossier Assembled ]
             |
             v
[ Step 13: Report Rendered on UI & Exported via jsPDF ]
```

---

## 10.2 Detailed Step-by-Step Breakdown

### Step 1: User Opens ABHERA
- The survivor navigates to `https://abhera.vercel.app` (or `http://localhost:5173` locally).
- The React application initializes `ConversationContext`. If no active session ID exists in `sessionStorage`, the frontend calls `POST /api/session/start`.
- The backend generates a secure anonymous session identifier (e.g., `SESS-8f12a4b7`) and returns `conversation_status: "ACTIVE"`.

### Step 2: User Enters an Incident Narrative
- The survivor navigates to `/chatbot` and types their experience in unconstrained natural language into the auto-resizing textarea.
- The user is free to write casually, emotionally, or formally. No specific legal jargon or checkboxes are required.

### Step 3: Frontend Sends the Request
- When the user presses Enter or clicks the Send button, `frontend/src/services/api.js` fires an asynchronous HTTP `POST` request to `/api/chat/message` with JSON body:
  ```json
  {
    "session_id": "SESS-8f12a4b7",
    "message": "My supervisor at work keeps sending me inappropriate messages on WhatsApp and threatened my job if I report him."
  }
  ```
- React sets `loading: true`, rendering a subtle pulsing spinner.

### Step 4: Backend Receives and Validates the Request
- FastAPI receives the HTTP request on port 8000.
- Pydantic validates `ChatMessageRequest`. If `session_id` or `message` is empty, an immediate `400 Bad Request` is returned.
- FastAPI injects a database session (`Depends(get_db)`) and delegates to `ChatService.process_message()`.

### Step 5: BERT Multi-Label Incident Classification
- `ChatService` invokes the in-memory singleton `BERTIncidentPredictor.predict(narrative_text)`.
- The text is tokenized into WordPiece IDs and passed through the 12 encoder layers of `models/bert_multilabel_experiment_v2/`.
- Independent Sigmoids compute probabilities for all 6 classes:
  - `DV`: 0.04
  - `SH`: 0.92 ($\ge 0.50 \implies \mathbf{True}$)
  - `ST`: 0.12
  - `CA`: 0.81 ($\ge 0.45 \implies \mathbf{True}$)
  - `WH`: 0.89 ($\ge 0.29 \implies \mathbf{True}$)
  - `OV`: 0.02
- Predicted Categories: **`Sexual Harassment (SH)`**, **`Cyber Abuse (CA)`**, and **`Workplace Harassment (WH)`**.

### Step 6: BERT NER Entity Extraction
- `ChatService` invokes `NEREntityPredictor.predict(narrative_text)`.
- The token classifier predicts 13 BIO tags across tokens.
- Offset mapping slices character spans from the raw text:
  - `PERP_REL`: `["supervisor"]`
  - `LOCATION`: `["at work"]`
  - `PLATFORM`: `["WhatsApp"]`

### Step 7: Relevant Follow-Up Questions Selected
- The `DynamicQuestionEngine` evaluates required slots for `WH`, `SH`, and `CA`:
  - Required dimensions: `PERP_REL`, `LOCATION`, `PLATFORM`, `TIME_FREQ`, `INCIDENT_DETAILS`, `SAFETY_STATUS`.
  - Already Known from NER: `PERP_REL` ("supervisor"), `LOCATION` ("at work"), `PLATFORM` ("WhatsApp").
  - Evaluated Missing Slots: `SAFETY_STATUS` (Priority 100), `TIME_FREQ` (Priority 85).
- The Question Selector picks the highest priority missing slot: `SAFETY_STATUS`.
- The bot replies empathetically, acknowledges the workplace incident, and asks:
  > *"Are you currently in immediate physical danger, or do you need emergency support right now?"*
- Session state updates to `conversation_status: "QUESTIONING"`.

### Step 8: User Answers the Follow-Up Question
- The survivor types: *"No, I am physically safe right now, but he has been doing this for the past month."*
- The answer is sent back via `POST /api/chat/message`.
- NER extracts `TIME_FREQ`: `["for the past month"]`.

### Step 9: Data Stored in PostgreSQL
- The answer is saved into `submission_question_answers` table.
- All required informational slots are now fulfilled!
- The state machine sets `conversation_status: "COMPLETED"`.

### Step 10: Relevant Legal Guidance Retrieved
- The `AntiHallucinationService` queries the PostgreSQL `laws` table for verified statutes matching `SH`, `WH`, and `CA`:
  - **POSH Act 2013, Section 3 & 4**: Prohibition of sexual harassment at workplace, constitution of Internal Complaints Committee.
  - **Bharatiya Nyaya Sanhita (BNS 2023), Section 75**: Sexual harassment and punishment (1 to 3 years imprisonment).
  - **Information Technology Act 2000, Section 66E / 67**: Transmission of obscene material in electronic form.
- The anti-hallucination layer verifies each row against the database.

### Step 11: Emergency Support Information Retrieved
- The system queries `support_services` matching the incident categories:
  - National Emergency Hotline: `112`
  - National Women Helpline: `181`
  - National Cyber Crime Reporting Portal: `https://cybercrime.gov.in` (Toll-Free `1930`)
  - National Commission for Women (NCW) Grievance Cell: `7827170170`

### Step 12: Final Structured Report Compiled
- `ReportGeneratorService.generate_report()` compiles the data into the 10-section JSON schema.
- The record is persisted into the `reports` table in PostgreSQL.

### Step 13: Report Rendered on UI and Exported to PDF
- The user is automatically transitioned to `/report`.
- The frontend renders an interactive, tabbed dashboard with legal citations and helpline cards.
- The user clicks **"Download Official PDF Report"**, and `jsPDF` compiles a vector PDF directly in the browser.

---

## 10.3 Complete Fictional Worked Example

### 1. Initial Survivor Narrative
```
"My husband and his parents have been physically beating me and demanding 5 lakhs in cash. Yesterday he threatened to throw me out of the house in Hyderabad."
```

### 2. Deep Learning Neural Output
- **BERT Classifier Logits & Probabilities**:
  - `DV`: $p = 0.9821 \ge 0.73 \implies \mathbf{True}$
  - `OV`: $p = 0.8140 \ge 0.63 \implies \mathbf{True}$
  - `SH`: $p = 0.0512 < 0.50 \implies False$
  - `ST`: $p = 0.0210 < 0.79 \implies False$
  - `CA`: $p = 0.0105 < 0.45 \implies False$
  - `WH`: $p = 0.0080 < 0.29 \implies False$
  - **Active Labels**: `["Domestic Violence", "Other Violence"]`
- **NER Extracted Entities**:
  - `PERP_REL`: `["husband", "his parents"]`
  - `LOCATION`: `["house in Hyderabad"]`
  - `EVIDENCE`: `["demanding 5 lakhs in cash"]`

### 3. Missing Slots & Dynamic Follow-up Turn
- Required for Domestic Violence: `PERP_REL`, `INCIDENT_DETAILS`, `SAFETY_STATUS`.
- Missing: `SAFETY_STATUS`.
- Bot Inquiry: *"Are you currently in immediate physical danger, or do you need emergency support right now?"*
- User Response: *"I am staying with my sister tonight so I am safe, but I want to take legal action."*
- Dynamic Question Engine resolves `SAFETY_STATUS: "safe tonight"`. State $\to$ `COMPLETED`.

### 4. Deterministic Legal Provisions Retrieved from PostgreSQL `laws` Table:
1. **Protection of Women from Domestic Violence Act, 2005 (PWDVA)**:
   - *Section 3*: Definition of domestic violence (physical, verbal, emotional, and economic abuse).
   - *Section 18 & 19*: Protection orders and residence orders prohibiting dispossession of shared household.
2. **Bharatiya Nyaya Sanhita, 2023 (BNS)**:
   - *Section 85*: Cruelty by husband or relatives of husband (imprisonment up to 3 years and fine).
   - *Section 86*: Definition of cruelty including harassment to coerce unlawful property/dowry demands.
3. **Dowry Prohibition Act, 1961**:
   - *Section 3 & 4*: Penalty for giving, taking, or demanding dowry.

### 5. Verified Support Services Retrieved from `support_services` Table:
- **Emergency Police**: `112` (Immediate emergency assistance)
- **Women Helpline (All India)**: `181` (Toll-free 24/7 crisis support and shelter referral)
- **Sakhi One Stop Centre (Telangana)**: Integrated medical, legal, and counseling center.
'''
