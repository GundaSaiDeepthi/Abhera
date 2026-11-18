"""
Module for Chapter 3: Complete Project Architecture
"""

def get_chapter_03():
    return r'''# CHAPTER 3: Complete Project Architecture

---

## 3.1 Overall System Architecture

ABHERA is designed around a **multi-tier, decoupled, neuro-symbolic architecture**. 

"Neuro-symbolic" means the system strictly divides responsibilities between two distinct paradigms of computer science:
1. **Neural Perception (Deep Learning)**: Probabilistic understanding of unstructured human speech using transformer neural networks (BERT).
2. **Symbolic Determinism (Relational Logic & Rules)**: 100% deterministic, rule-governed knowledge retrieval from relational database tables (PostgreSQL) guarded by an Anti-Hallucination verification layer.

```
+===================================================================================================+
|                                  ABHERA OVERALL SYSTEM ARCHITECTURE                               |
+===================================================================================================+

  [ USER / SURVIVOR ]
         |
         | (1) Natural Language Intake & Questions (HTTPS)
         v
+---------------------------------------------------------------------------------------------------+
| TIER 1: FRONTEND SINGLE-PAGE APPLICATION (React 18 + Vite)                                        |
| Location: frontend/src/                                                                           |
| Hosting: Vercel Global Edge Network                                                               |
|                                                                                                   |
|  [ Home View ] ---> [ Guided Chatbot View ] ---> [ Processing Spinner ] ---> [ Interactive Report]|
|         ^                     |                           |                          |            |
|         |                     |                           |                          v            |
|  [ Navbar / Emblem ]   [ ChatInput / Msgs ]        [ Polling / State ]        [ jsPDF Compiler ]  |
|                               |                                                      |            |
|                               +-------------------+----------------------------------+            |
|                                                   |                                               |
|                                                   v                                               |
|                                    [ ConversationContext (State) ]                                |
|                                    [ api.js (Axios REST Client)  ]                                |
+---------------------------------------------------------------------------------------------------+
                                                    |
                                                    | (2) JSON REST API Requests via HTTPS
                                                    |     (Localhost:8000 or Cloudflare Tunnel)
                                                    v
+---------------------------------------------------------------------------------------------------+
| TIER 2: BACKEND APPLICATION SERVER (FastAPI + Starlette + Pydantic v2)                            |
| Location: backend/app/                                                                            |
| Execution: Asynchronous Uvicorn ASGI Server                                                       |
|                                                                                                   |
|  [ CORS Middleware ] -> [ Request Validation (Pydantic) ] -> [ APIRouters: chat, session, report] |
|                                       |                                                           |
|                                       v                                                           |
|                          [ ChatService Orchestrator ]                                             |
|                                       |                                                           |
|         +-----------------------------+-----------------------------+                             |
|         |                                                           |                             |
|         v                                                           v                             |
|  [ Dynamic Question Engine ]                                 [ Anti-Hallucination Service ]       |
|  - rules.py (Slot definitions)                               - Provenance assert against DB       |
|  - question_selector.py (Priority triage)                    - Rejects unverified statutory laws  |
|  - dynamic_generator.py (Contextual phrasing)                - Fallback guardrails                |
+---------------------------------------------------------------------------------------------------+
         |                                                           |
         | (3) Python ML In-Memory Inference                         | (4) SQLAlchemy 2.0 ORM Queries
         v                                                           v
+----------------------------------------------+  +-------------------------------------------------+
| TIER 3: DEEP LEARNING INFERENCE RUNTIME      |  | TIER 4: RELATIONAL KNOWLEDGE BASE               |
| Location: ml/inference/                      |  | Location: PostgreSQL 16 (womens_safety_db)       |
| Framework: PyTorch 2.14 + Hugging Face       |  |                                                 |
|                                              |  |  [ laws Table ] (32 Audited Provisions)         |
|  [ BERT Multi-Label Classifier ]             |  |  - BNS 2023, IT Act 2000, POSH 2013, PWDVA 2005 |
|  - models/bert_multilabel_experiment_v2/     |  |                                                 |
|  - 6 Classes: DV, SH, ST, CA, WH, OV         |  |  [ support_services Table ] (12 Helplines)      |
|  - pos_weight BCE loss + Optimal Thresholds  |  |  - National 112, 181, 1090, Cybercrime, NCW    |
|                                              |  |                                                 |
|  [ BERT Token Classifier (NER) ]             |  |  [ Transactional Tables ]                       |
|  - models/ner/                               |  |  - users, sessions, incident_submissions       |
|  - 13 BIO Tags across 6 Entity Dimensions    |  |  - conversation_messages, questions, reports   |
|  - WordPiece Sub-word Offset Alignment       |  +-------------------------------------------------+
+----------------------------------------------+
```

---

## 3.2 Frontend Architecture

The frontend is structured as a **Single Page Application (SPA)** using React 18 and Vite. It provides zero-friction anonymous access and operates as an intelligent client state machine.

```mermaid
flowchart TD
    subgraph ClientApp["React 18 Single-Page Application (frontend/src)"]
        Index["index.html (Mount Root)"] --> Main["main.jsx"]
        Main --> App["App.jsx (Router & Layout)"]
        
        subgraph StateManagement["Global Context Provider"]
            Ctx["ConversationContext.jsx"]
            Store["sessionStorage (Session & Submission UUIDs)"]
            Ctx <--> Store
        end
        
        App --> Ctx
        
        subgraph ViewPages["Application Pages (pages/)"]
            Home["Home.jsx (Hero & Educational Guide)"]
            Chat["Chatbot.jsx (Conversational Intake)"]
            Proc["Processing.jsx (Analysis Transition)"]
            Rep["Report.jsx (Evidentiary Dossier)"]
        end
        
        Ctx --> Home
        Ctx --> Chat
        Ctx --> Proc
        Ctx --> Rep
        
        subgraph UIComponents["Reusable Components (components/)"]
            Nav["Navbar.jsx"]
            Inp["ChatInput.jsx"]
            Msg["ChatMessage.jsx"]
            Ldr["LoadingIndicator.jsx"]
            Emb["AbheraEmblem.jsx"]
        end
        
        Chat --> Inp
        Chat --> Msg
        Chat --> Ldr
        Rep --> Emb
        
        subgraph Services["Network & PDF Layer"]
            Api["services/api.js (Axios HTTP Client)"]
            Pdf["jsPDF Vector Compiler (Client-Side)"]
        end
        
        Chat --> Api
        Rep --> Api
        Rep --> Pdf
    end
```

### Key Architectural Strengths of the Frontend:
1. **Zero-PII Storage**: No personal names, emails, passwords, or phone numbers are ever stored in React state.
2. **Session Resilience**: Active sessions are synchronized with the browser's native `sessionStorage`. If the survivor accidentally refreshes the page, their active conversation history and session ID remain intact.
3. **In-Memory PDF Compilation**: Unlike legacy systems that generate PDFs on a remote server (which creates a privacy risk of sensitive files sitting on server hard drives), ABHERA compiles the entire PDF directly inside browser RAM using `jsPDF`.

---

## 3.3 Backend Architecture

The backend is built using **FastAPI**, an asynchronous ASGI Python framework designed for high-concurrency API microservices.

```mermaid
flowchart TD
    subgraph FastAPIBackend["FastAPI Backend Architecture (backend/app)"]
        Entry["main.py (ASGI Application Entry)"]
        Config["config.py (Settings & Environment)"]
        Database["database.py (SQLAlchemy Engine & SessionLocal)"]
        
        Entry --> Config
        Entry --> Database
        
        subgraph Middleware["HTTP Middleware Layer"]
            CORS["CORSMiddleware (Origins: localhost, Vercel)"]
        end
        Entry --> CORS
        
        subgraph Routers["REST API Routers (routes/)"]
            R_Health["health.py (/health)"]
            R_Session["session.py (/api/session)"]
            R_Chat["chat.py (/api/chat)"]
            R_Report["report.py (/api/report)"]
        end
        CORS --> Routers
        
        subgraph Schemas["Pydantic v2 Request/Response Validation (schemas/)"]
            S_Chat["chat.py (ChatMessageRequest, ChatMessageResponse)"]
            S_Session["session.py (SessionStartRequest, SessionResponse)"]
            S_Report["report.py (ReportGenerateRequest, ReportResponse)"]
        end
        Routers --> Schemas
        
        subgraph Services["Core Business Logic Services (services/)"]
            Svc_Chat["ChatService (Pipeline Orchestrator)"]
            Svc_Session["SessionService (Lifecycle Manager)"]
            Svc_Report["ReportGeneratorService (10-Section Compiler)"]
            Svc_AntiH["AntiHallucinationService (DB Verification)"]
            Svc_Legal["LegalMappingService (Statutory Retrieval)"]
            Svc_Support["SupportMappingService (Emergency Routing)"]
        end
        Routers --> Services
        
        subgraph QuestionEngine["Dynamic Question Engine (question_engine/)"]
            QE_Engine["engine.py (State Evaluator)"]
            QE_Rules["rules.py (Category Slot Definitions)"]
            QE_Selector["question_selector.py (Priority Triage)"]
            QE_Dynamic["dynamic_generator.py (Contextual Phrasing)"]
        end
        Svc_Chat --> QuestionEngine
    end
```

---

## 3.4 Machine Learning Architecture

ABHERA contains two independent deep learning transformer models that run sequentially inside the backend process:

```
[ Unstructured Incident Narrative Text ]
                 |
                 +-----------------------------------+
                 |                                   |
                 v                                   v
+----------------------------------+  +----------------------------------+
| MODEL 1: MULTI-LABEL CLASSIFIER  |  | MODEL 2: TOKEN CLASSIFIER (NER)  |
| Path: models/bert_multilabel_    |  | Path: models/ner/                |
|       experiment_v2/             |  |                                  |
| Base: bert-base-uncased          |  | Base: bert-base-uncased          |
| Task: Sequence Classification    |  | Task: Token Sequence Labeling    |
| Max Length: 128 tokens           |  | Max Length: 256 tokens           |
|                                  |  |                                  |
| Input: Narrative String          |  | Input: Narrative String          |
|      |                           |  |      |                           |
|      v                           |  |      v                           |
| [ 12 Transformer Encoder Layers] |  | [ 12 Transformer Encoder Layers] |
|      |                           |  |      |                           |
|      v                           |  |      v                           |
| [ [CLS] Token 768-dim Vector ]   |  | [ All Token 768-dim Vectors ]    |
|      |                           |  |      |                           |
|      v                           |  |      v                           |
| [ Linear Projection Head (768->6)]| | [ Linear Projection (768->13) ]  |
|      |                           |  |      |                           |
|      v                           |  |      v                           |
| [ 6 Independent Sigmoids ]       |  | [ 13-Tag BIO Sequence ]          |
|      |                           |  |      |                           |
|      v                           |  |      v                           |
| [ Per-Class Optimal Thresholds ] |  | [ Sub-Word Offset Alignment ]    |
| DV: 0.73 | SH: 0.50 | ST: 0.79   |  | Spans: PERP_REL, LOCATION,       |
| CA: 0.45 | WH: 0.29 | OV: 0.63   |  |        TIME_FREQ, PLATFORM       |
+----------------------------------+  +----------------------------------+
                 |                                   |
                 +-----------------+-----------------+
                                   |
                                   v
             [ Structured Multi-Label Output & Entity Dict ]
```

---

## 3.5 Database Architecture

The database architecture is centered on strict normalization, referential integrity, and forensic auditability.

```mermaid
erDiagram
    users ||--o{ sessions : "owns"
    sessions ||--o| incident_submissions : "initializes"
    incident_submissions ||--o{ conversation_messages : "records"
    incident_submissions ||--o{ predictions : "generates"
    incident_submissions ||--o{ entities : "extracts"
    incident_submissions ||--o{ submission_question_answers : "answers"
    questions ||--o{ submission_question_answers : "answered_in"
    incident_submissions ||--o| reports : "compiles"
    
    users {
        int id PK
        string user_guid
        boolean is_anonymous
        datetime created_at
    }
    
    sessions {
        int id PK
        string session_id UK
        string submission_id
        int user_id FK
        json messages
        json answers
        json predicted_labels
        json entities
        json current_question
        string conversation_status
        datetime created_at
        datetime updated_at
    }
    
    incident_submissions {
        int id PK
        string submission_id UK
        string session_id FK
        text narrative_text
        string state
        string district
        json predicted_labels
        json extracted_entities
        boolean report_generated
        datetime timestamp
    }
    
    laws {
        int id PK
        string act_name
        string section_number
        text section_text
        string applicable_label
        datetime created_at
    }
    
    support_services {
        int id PK
        string service_type
        string name
        string contact_number
        string state
        string district
        string applicable_label
        datetime created_at
    }
    
    reports {
        int id PK
        string submission_id FK,UK
        json report_data
        datetime created_at
        datetime updated_at
    }
```

---

## 3.6 Communication Between Components

1. **Frontend to Backend**: Communicates exclusively over HTTP/HTTPS using RESTful JSON payloads. Cross-Origin Resource Sharing (CORS) is explicitly configured in `backend/app/config.py` to allow requests from authorized frontend domains (`http://localhost:5173`, `https://abhera.vercel.app`).
2. **Backend to ML Models**: Models are loaded directly into server RAM during initial application startup using singleton patterns. Inference calls are executed in-memory as direct Python function invocations without network latency.
3. **Backend to PostgreSQL**: Communicates over TCP port 5432 using SQLAlchemy connection pooling. Connection strings are parameterized via environment variables (`DATABASE_URL`).
4. **Backend to Anti-Hallucination Layer**: Operates as an internal gating function that queries PostgreSQL `laws` and `support_services` before any payload is returned to the router.

---

## 3.7 Data Flow Diagrams (DFD)

### DFD Level 0 (Context Diagram)
The Level 0 context diagram shows the entire ABHERA system as a single black-box process interacting with external entities:

```
                  +-----------------------------------+
                  |        Survivor / Citizen         |
                  +-----------------------------------+
                     |                             ^
                     | (1) Narrative & Answers     | (4) Questions, Legal Info,
                     v                             |     PDF Dossier
                  +-----------------------------------+
                  |                                   |
                  |          ABHERA SYSTEM            |
                  |                                   |
                  +-----------------------------------+
                     ^                             |
                     | (2) Verified Laws &         | (3) Query Labels
                     |     Helpline Data           |     & Locations
                     v                             v
                  +-----------------------------------+
                  |       PostgreSQL Knowledge        |
                  |        Base (BNS / IT Act)        |
                  +-----------------------------------+
```

### DFD Level 1 (Macro Process Flow)
The Level 1 diagram decomposes ABHERA into its five primary functional subsystems:

```
[Survivor] ---> (Process 1.0: Session & Intake) ---> [UserSession Store]
                     |
                     v (Raw Narrative)
                (Process 2.0: Neural Inference) <---> [BERT Models in RAM]
                     |
                     v (Labels & Entities)
                (Process 3.0: Slot Evaluation) <---> [Question Bank DB]
                     |
                     v (Missing Slots / Follow-up Qs)
                (Process 4.0: Knowledge Retrieval) <---> [laws & support_services DB]
                     |
                     v (Verified Statutes & Support)
                (Process 5.0: Dossier Compilation) ---> [Immutable Report PDF] ---> [Survivor]
```

### DFD Level 2 (Detailed Neural Inference & Anti-Hallucination Gating)
```
[Raw Narrative Text]
        |
        +---> [WordPiece Tokenizer] ---> [Input IDs & Attention Mask]
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                                                 |
                   v                                                                 v
        [BERT Multi-Label Head]                                            [BERT Token Classification Head]
                   |                                                                 |
                   v (Logits: DV, SH, ST, CA, WH, OV)                               v (BIO Token Logits)
        [Sigmoid Activation Layer]                                         [Argmax Tag Selection]
                   |                                                                 |
                   v (Probabilities 0.0 - 1.0)                                       v (BIO Tags: B-PERP, I-PERP)
        [Threshold Tuning Engine]                                          [Character Offset Aligner]
                   |                                                                 |
                   v (Predicted Labels: e.g. ST, CA)                                 v (Spans: "ex-husband", "Instagram")
                   |                                                                 |
                   +--------------------------------+--------------------------------+
                                                    |
                                                    v
                                    [Anti-Hallucination Gating Engine]
                                                    |
                                                    +---> Query: `SELECT * FROM laws WHERE applicable_label IN ('ST', 'CA')`
                                                    |
                                                    +---> Assert: Verified Rows Found?
                                                                 |
                                              +------------------+------------------+
                                              | YES                                 | NO
                                              v                                     v
                                    [Append Verified Statutes]            [Return Mandatory Safe Fallback]
```

---

## 3.8 Sequence Diagram (End-to-End Chat & Intake)

This sequence diagram illustrates the exact chronological order of function calls and network messages during a user turn:

```mermaid
sequenceDiagram
    autonumber
    actor Survivor as Survivor (Browser)
    participant UI as React Frontend (Chatbot.jsx)
    participant API as FastAPI Router (chat.py)
    participant Svc as ChatService Orchestrator
    participant BERT as BERTIncidentPredictor
    participant NER as NEREntityPredictor
    participant QE as Dynamic Question Engine
    participant DB as PostgreSQL 16 (laws / sessions)
    participant AH as AntiHallucinationService

    Survivor->>UI: Enters incident narrative
    UI->>API: POST /api/chat/message {session_id, message}
    API->>Svc: process_message(session_id, message)
    
    Note over Svc,DB: Step 1: Persist User Message
    Svc->>DB: INSERT INTO conversation_messages (sender="USER")
    
    Note over Svc,BERT: Step 2: Multi-Label Classification
    Svc->>BERT: predict(narrative_text)
    BERT-->>Svc: {"labels": ["ST", "CA"], "probabilities": {"ST": 0.88, "CA": 0.72}}
    
    Note over Svc,NER: Step 3: Named Entity Recognition
    Svc->>NER: predict(narrative_text)
    NER-->>Svc: [{"entity_type": "PERP_REL", "text": "ex-colleague"}, {"entity_type": "PLATFORM", "text": "WhatsApp"}]
    
    Note over Svc,QE: Step 4: Evaluate Missing Evidentiary Slots
    Svc->>QE: select_next_question(labels, known_entities)
    QE-->>Svc: {"question_id": "Q_TIME_FREQ_01", "text": "How long or frequently has this been happening?"}
    
    Note over Svc,AH: Step 5: Statutory Retrieval & Anti-Hallucination
    Svc->>AH: validate_payload(labels=["ST", "CA"])
    AH->>DB: SELECT * FROM laws WHERE applicable_label IN ('ST', 'CA')
    DB-->>AH: Verified rows (BNS Section 78, IT Act Section 66E)
    AH-->>Svc: Validated legal citations
    
    Svc->>DB: UPDATE sessions SET current_question=Q_TIME_FREQ_01, status="QUESTIONING"
    Svc-->>API: ChatMessageResponse
    API-->>UI: HTTP 200 OK (JSON)
    UI-->>Survivor: Displays empathetic response + Follow-up Question
```
'''
