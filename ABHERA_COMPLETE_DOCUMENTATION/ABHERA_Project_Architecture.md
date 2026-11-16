# ABHERA System Architecture & Technical Design Specification
**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**Document Type:** Master Architecture & Engineering Specification  
**Version:** 2.0.0 (Production Verified)  
**Author:** Technical Engineering & Verification Team  

---

## 1. Architectural Philosophy & Design Principles

ABHERA is architected as a **decoupled, multi-tier, privacy-preserving legal intake and assistance system**. Its architectural decisions are governed by five fundamental design tenets:

1. **Deterministic Legal Grounding (Anti-Hallucination by Design):**  
   Large Language Models (LLMs) frequently hallucinate statutory sections, penalties, and court jurisdictions. ABHERA strictly rejects generative LLM synthesis for legal counsel. Instead, neural models are confined exclusively to perceptual feature extraction (classification and token-level entity recognition), while legal recommendations are computed via deterministic relational queries against vetted PostgreSQL statutory tables.

2. **Survivor Privacy & Zero-Credential Ephemerality:**  
   Survivors of domestic violence or cyber abuse frequently face coercive phone snooping. ABHERA collects zero emails, zero passwords, and zero personal credentials. Sessions are anchored to pseudonymous client-side tokens (`sessionStorage`) that leave no browser history or persistent user profile.

3. **Asynchronous Micro-Service Separation:**  
   The system decouples the client-side Single Page Application (React 18 + Vite) from the backend API gateway (FastAPI + Uvicorn) and deep learning runtime (PyTorch CPU transformers). This separation ensures that compute-heavy transformer forward passes (~300–450ms) never block the responsive client UI.

4. **Trauma-Informed Conversational Slot-Filling:**  
   Unlike rigid linear legal questionnaires, ABHERA accepts unstructured natural language disclosures. An adaptive Dynamic Question Engine identifies evidentiary gaps (such as missing perpetrator relationship or physical location) and poses non-redundant, empathetic follow-up prompts before compiling the legal dossier.

5. **Client-Side Document Cryptography & Vector Compilation:**  
   Sensitive incident dossiers are rendered into PDF format directly inside the survivor's browser memory via `jsPDF`. No server-side document caching or headless browser instances (such as Puppeteer) are required, eliminating server-side file leakage vectors.

---

## 2. High-Level System Architecture Diagram

```
+=======================================================================================================+
|                                  ABHERA OVERALL SYSTEM ARCHITECTURE                                   |
+=======================================================================================================+

    +-----------------------------------------------------------------------------------------------+
    |                           CLIENT PRESENTATION TIER (React 18 + Vite)                          |
    |                                                                                               |
    |  +--------------------+   +--------------------+   +--------------------+   +---------------+ |
    |  |     Home.jsx       |   |    Chatbot.jsx     |   |   Processing.jsx   |   |  Report.jsx   | |
    |  | (Emergency Crisis) |   | (Intake & Dialogue)|   | (Model Progress)   |   | (Vector jsPDF)| |
    |  +--------------------+   +--------------------+   +--------------------+   +---------------+ |
    |                                     |                                                         |
    |                 Global State: ConversationContext.jsx <==> sessionStorage                     |
    |                                     |                                                         |
    |                 Central Axios Client: frontend/src/services/api.js (VITE_API_URL)             |
    +-------------------------------------|---------------------------------------------------------+
                                          |
                                          | JSON over HTTPS (RESTful API)
                                          v
    +-----------------------------------------------------------------------------------------------+
    |                         APPLICATION GATEWAY & LOGIC TIER (FastAPI ASGI)                       |
    |                                                                                               |
    |  FastAPI Middleware: CORSMiddleware (CORS_ORIGINS), Exception Handlers, Pydantic Validation   |
    |                                                                                               |
    |  REST API Routers:                                                                            |
    |  - routes/health.py   ==> Health checks and system diagnostics                                |
    |  - routes/session.py  ==> POST /api/session/start, GET /api/session/{session_id}              |
    |  - routes/chat.py     ==> POST /api/chat/message (Core Intake & Conversational Turns)         |
    |  - routes/report.py   ==> GET /api/report/{submission_id} (Dossier Retrieval)                |
    |                                     |                                                         |
    |  Service Orchestration Layer:       |                                                         |
    |  +----------------------------------+------------------------------------------------------+  |
    |  | ChatService.process_message()                                                           |  |
    |  |  ├── 1. Session & Submission Lifecycle Manager (SessionService)                         |  |
    |  |  ├── 2. Dual Deep Learning Inference Call (predict_bert.py + predict_ner.py)            |  |
    |  |  ├── 3. Dynamic Question Engine & Heuristic Priority Queue (engine.py)                  |  |
    |  |  ├── 4. Statutory Rule Mapping Engine (LegalMappingService)                             |  |
    |  |  ├── 5. 3-Tier Geospatial Support Service Matcher (SupportMappingService)               |  |
    |  |  ├── 6. Strict Provenance & Database Verification Layer (AntiHallucinationService)        |  |
    |  |  └── 7. 10-Section Dossier Compiler (ReportGeneratorService)                            |  |
    |  +-----------------------------------------------------------------------------------------+  |
    +-------------------------------------|---------------------------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
    +----------------------------------+     +------------------------------------------------------+
    |    NEURAL PERCEPTION WORKERS     |     |          STORAGE TIER: POSTGRESQL 16                 |
    |          (PyTorch CPU)           |     |         (SQLAlchemy 2.0 Connection Pool)             |
    |                                  |     |                                                      |
    |  1. BERT Multi-Label Classifier  |     |  Relational Tables (11 Active Normalized Schemas):   |
    |     - Model: bert-base-uncased   |     |  - users                 - questions                 |
    |     - 110M Params | 6 Epochs     |     |  - sessions              - submission_question_ans   |
    |     - Pos-Weighted BCE Loss      |     |  - incident_submissions  - laws (32 BNS/IT/POSH rows)|
    |     - Micro-F1: 98.09%           |     |  - conversation_messages - support_services (12 rows)|
    |     - Optimal Decision Thresholds|     |  - predictions           - reports (10-Section JSON) |
    |                                  |     |  - entities                                          |
    |  2. BERT Token Classifier (NER)  |     |                                                      |
    |     - Model: BertForTokenClass.  |     |  Deterministic Database Seeders & Importers:         |
    |     - 13 BIO Tag Ontology        |     |  - init_db.py (Table Schema Bootstrap)               |
    |     - Fast WordPiece Offsets     |     |  - seed_questions.py (Interview Prompts)             |
    |     - Micro-F1: 99.24%           |     |  - data_importer.py (DatasetImporter CSV Loader)     |
    +----------------------------------+     +------------------------------------------------------+
```

---

## 3. Detailed Request-Response Lifecycle Trace

Every user interaction follows an immutable, verified 7-stage processing lifecycle:

```
Survivor Client                 FastAPI Gateway                ChatService              ML & DB Engines
       |                               |                            |                          |
       | 1. POST /api/chat/message     |                            |                          |
       |------------------------------>| 2. Pydantic Validate       |                          |
       |                               |--------------------------->| 3. Fetch/Create Session  |
       |                               |                            |------------------------->| (DB Query)
       |                               |                            |                          |
       |                               |                            | 4. Run BERT Classifier   |
       |                               |                            |------------------------->| (Forward Pass)
       |                               |                            |                          |
       |                               |                            | 5. Run BERT NER Extractor|
       |                               |                            |------------------------->| (Subword Offsets)
       |                               |                            |                          |
       |                               |                            | 6. Evaluate Question Slot|
       |                               |                            |    Queue & Anti-Looping  |
       |                               |                            |                          |
       |                               |                            | 7. Query Laws & Helplines|
       |                               |                            |------------------------->| (PostgreSQL SELECT)
       |                               |                            |                          |
       |                               | 8. Serialize JSON Response |<-------------------------|
       |<------------------------------|                            |                          |
       | 9. Update UI Badges & Prompt  |                            |                          |
```

### Stage 1: Request Packaging & Dispatch
The survivor enters an incident narrative or answers a follow-up prompt. `frontend/src/services/api.js` packages the payload:
```json
{
  "session_id": "8f3b2e10-74a9-4b61-9c12-3456789abcde",
  "message": "My husband hit me yesterday and threatened to kill me if I go to the police."
}
```

### Stage 2: Ingress Validation & Router Dispatch
FastAPI routes the payload to `backend/app/routes/chat.py`. The request body is validated against `ChatMessageRequest`. If `session_id` or `message` is empty or malformed, the router aborts immediately with `HTTP 400 Bad Request`.

### Stage 3: Conversational Intake Orchestration (`ChatService.process_message`)
`ChatService` in `backend/app/services/chat_service.py` coordinates execution:
1. Queries `sessions` table using `SessionService.get_session()`.
2. Locates or creates the corresponding `incident_submissions` record.
3. Appends the turn to `conversation_messages`.
4. Appends incoming user text to the cumulative narrative.

### Stage 4: Dual Neural Feature Extraction
1. **BERT Multi-Label Classification:**  
   `BERTIncidentPredictor.predict()` tokenizes narrative text to 128 WordPiece tokens. Forward pass produces raw logits transformed via independent sigmoids:
   $$\hat{p}_c = rac{1}{1 + e^{-z_c}} \quad orall c \in \{	ext{DV, SH, ST, CA, WH, OV}\}$$
   Scores are evaluated against calibrated validation thresholds (`DV: 0.73`, `SH: 0.50`, `ST: 0.79`, `CA: 0.45`, `WH: 0.29`, `OV: 0.63`). Positive labels are logged to `predictions`.
2. **BERT Named Entity Recognition:**  
   `NEREntityPredictor.predict()` performs dense token classification across 13 BIO tags. WordPiece tokens are re-aligned to raw narrative character offsets, extracting evidentiary spans (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`) and persisting them to `entities`.

### Stage 5: Dynamic Question Prioritization
`process_submission()` in `backend/app/question_engine/engine.py` evaluates slot fulfillment across 6 canonical dimensions:
$$	ext{Missing Slots} = 	ext{Category Mandated Slots} \setminus 	ext{Extracted Entity Slots}$$
Follow-up prompts are selected strictly following canonical legal priority:
$$	ext{SAFETY\_STATUS} \succ 	ext{PERP\_REL} \succ 	ext{LOCATION} \succ 	ext{TIME\_FREQ} \succ 	ext{PLATFORM} \succ 	ext{INCIDENT\_DETAILS}$$
`dynamic_generator.py` personalizes question phrasing based on extracted context (e.g., referencing "your husband").

### Stage 6: Deterministic Legal & Support Retrieval
1. `LegalMappingService` queries PostgreSQL `laws` table, filtering records where `applicable_label` matches any predicted class.
2. `SupportMappingService` queries `support_services`, applying 3-tier geospatial scoring (+300 District, +200 State, +100 National Fallback).
3. `AntiHallucinationService` asserts primary-key integrity and verbatim text presence against database records.

### Stage 7: Dossier Serialization & Client-Side PDF Compilation
`ReportGeneratorService` constructs the immutable 10-section JSON schema and commits it to `reports`. In `frontend/src/pages/Report.jsx`, `jsPDF` draws vector headers, gold accent rules, category badges, and paginated legal provisions directly onto an in-browser canvas.

---

## 4. Data Flow Architecture (DFD Hierarchy)

### Level 0 Context Diagram
```
+------------------+         Incident Narrative & Answers         +-------------------+
|                  | -------------------------------------------> |                   |
|                  |                                              |                   |
|     Survivor     | <------------------------------------------- |      ABHERA       |
|     Browser      |         Dynamic Prompts & Legal Dossier      |   Legal Safety    |
|                  |                                              |      System       |
+------------------+                                              +-------------------+
                                                                            |
                                                                            | Verified Queries
                                                                            v
                                                                  +-------------------+
                                                                  |   PostgreSQL 16   |
                                                                  |  (Laws & Support) |
                                                                  +-------------------+
```

### Level 1 Decomposition Diagram
```
              [ Survivor User Input ]
                         |
                         v
      +-------------------------------------+
      | 1.0 Conversational Session Manager  | <=======> ( sessions & submissions )
      +-------------------------------------+
                         |
                         v
      +-------------------------------------+
      | 2.0 Dual Deep Transformer Evaluator | <=======> [ models/safetensors ]
      +-------------------------------------+
            |                         |
      (Labels)                  (Entities)
            v                         v
      +-------------------------------------+
      | 3.0 Dynamic Follow-Up Engine        | <=======> ( questions & answers )
      +-------------------------------------+
                         |
                         v
      +-------------------------------------+
      | 4.0 Legal & Support Knowledge Match | <=======> ( laws & support_services )
      +-------------------------------------+
                         |
                         v
      +-------------------------------------+
      | 5.0 Anti-Hallucination Guardrail    |
      +-------------------------------------+
                         |
                         v
      +-------------------------------------+
      | 6.0 10-Section Dossier Compiler     | <=======> ( reports table )
      +-------------------------------------+
                         |
                         v
              [ In-Browser Vector PDF ]
```

---

## 5. Architectural Component Inventory

| Component ID | Module / File Path | Architectural Role | Core Technical Stack |
| :--- | :--- | :--- | :--- |
| **COMP-01** | `frontend/src/App.jsx` | Client View Router & Shell | React 18, React-Router-DOM |
| **COMP-02** | `frontend/src/context/ConversationContext.jsx` | Global Client State Provider | React Context, sessionStorage |
| **COMP-03** | `frontend/src/services/api.js` | REST Communication Layer | Axios, Promises |
| **COMP-04** | `frontend/src/pages/Report.jsx` | Dossier Viewer & PDF Compiler | jsPDF, Lucide-React |
| **COMP-05** | `backend/app/main.py` | ASGI Application Bootstrap | FastAPI, Uvicorn, CORSMiddleware |
| **COMP-06** | `backend/app/config.py` | System Configuration Registry | Pydantic BaseSettings, dotenv |
| **COMP-07** | `backend/app/database.py` | Relational Persistence Gateway | SQLAlchemy 2.0, Connection Pool |
| **COMP-08** | `backend/app/services/chat_service.py` | Business Logic Orchestrator | Python 3.10+, Transaction Scopes |
| **COMP-09** | `ml/inference/predict_bert.py` | Incident Multi-Label Classifier | PyTorch, Transformers, Sigmoids |
| **COMP-10** | `ml/inference/predict_ner.py` | Named Entity Token Classifier | PyTorch, WordPiece Offset Mapping |
| **COMP-11** | `backend/app/question_engine/engine.py` | Adaptive Interview Machine | Priority Queues, Slot Heuristics |
| **COMP-12** | `backend/app/services/legal_mapping.py` | Statutory Knowledge Retriev. | PostgreSQL Query Filters |
| **COMP-13** | `backend/app/services/support_mapping.py` | 3-Tier Support Geo-Ranker | Regex Matchers, Scoring Weights |
| **COMP-14** | `backend/app/services/anti_hallucination.py` | Provenance Verification Gate | Relational Foreign-Key Checks |
| **COMP-15** | `backend/app/services/report_generator.py` | 10-Section JSON Schema Builder | Python Dictionaries, ISO Datetimes |
