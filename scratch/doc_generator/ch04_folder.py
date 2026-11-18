"""
Module for Chapter 4: Complete Folder Structure
"""

def get_chapter_04():
    return r'''# CHAPTER 4: Complete Folder Structure

---

## 4.1 Master Project Directory Tree

Below is the verified, empirical folder structure of the ABHERA project located at `D:\\Abhera(Mini)`:

```
D:\Abhera(Mini)
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── main.py
│   │   ├── db/
│   │   │   ├── init_db.py
│   │   │   ├── migrate_questions_table.py
│   │   │   ├── migrate_sessions_table.py
│   │   │   └── seed_questions.py
│   │   ├── models/
│   │   │   ├── conversation_message.py
│   │   │   ├── entity.py
│   │   │   ├── incident_submission.py
│   │   │   ├── law.py
│   │   │   ├── prediction.py
│   │   │   ├── question.py
│   │   │   ├── report.py
│   │   │   ├── session.py
│   │   │   ├── support_service.py
│   │   │   ├── user.py
│   │   │   └── __init__.py
│   │   ├── question_engine/
│   │   │   ├── dynamic_generator.py
│   │   │   ├── engine.py
│   │   │   ├── question_selector.py
│   │   │   └── rules.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── health.py
│   │   │   ├── report.py
│   │   │   └── session.py
│   │   ├── schemas/
│   │   │   ├── chat.py
│   │   │   ├── report.py
│   │   │   └── session.py
│   │   └── services/
│   │       ├── anti_hallucination.py
│   │       ├── chat_service.py
│   │       ├── data_importer.py
│   │       ├── legal_mapping.py
│   │       ├── report_generator.py
│   │       ├── session_service.py
│   │       └── support_mapping.py
│   ├── tests/
│   │   ├── test_chat_api.py
│   │   ├── test_dynamic_question_generation.py
│   │   ├── test_fpm_incident.py
│   │   ├── test_hallucination.py
│   │   ├── test_legal_mapping.py
│   │   ├── test_multilabel_classification.py
│   │   ├── test_question_engine.py
│   │   ├── test_report_api.py
│   │   ├── test_report_generator.py
│   │   ├── test_session_api.py
│   │   └── test_support_mapping.py
│   ├── venv/
│   ├── .env
│   ├── .env.example
│   ├── README.md
│   └── requirements.txt
├── data/
│   ├── laws.csv
│   ├── support_services.csv
│   └── womens_safety_dataset.csv
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AbheraEmblem.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   ├── ChatMessage.jsx
│   │   │   ├── HeroIllustration.jsx
│   │   │   ├── LoadingIndicator.jsx
│   │   │   └── Navbar.jsx
│   │   ├── context/
│   │   │   └── ConversationContext.jsx
│   │   ├── pages/
│   │   │   ├── Chatbot.jsx
│   │   │   ├── Home.jsx
│   │   │   ├── Processing.jsx
│   │   │   └── Report.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── evaluation/
│   │   ├── FINAL_EVALUATION_RESULTS.json
│   │   └── results/
│   ├── inference/
│   │   ├── predict_bert.py
│   │   ├── predict_ner.py
│   │   └── test_predictor.py
│   ├── ner/
│   │   ├── prepare_ner_dataset.py
│   │   ├── train_ner.py
│   │   └── evaluate_ner.py
│   ├── preprocessing/
│   │   ├── prepare_classification.py
│   │   ├── train_data.csv
│   │   ├── val_data.csv
│   │   └── test_data.csv
│   └── training/
│       ├── train_bert.py
│       └── train_bert_experiment_v2.py
├── models/
│   ├── bert_multilabel/               # Legacy v1 checkpoint
│   ├── bert_multilabel_experiment_v2/ # Production Promoted Model
│   │   ├── config.json
│   │   ├── label_mapping.json
│   │   ├── model.safetensors
│   │   ├── optimal_thresholds.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── training_config.json
│   └── ner/                           # Production NER Token Classifier
│       ├── config.json
│       ├── label_mapping.json
│       ├── model.safetensors
│       ├── test_metrics.json
│       ├── tokenizer.json
│       └── training_config.json
├── documentation/
│   ├── ABHERA_Complete_Project_Guide_and_Viva.md
│   └── ABHERA_Complete_Project_Guide_and_Viva.pdf
├── .env.example
├── .gitignore
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## 4.2 Detailed Analysis of Key Folders and Files

### 1. `backend/app/main.py`
- **Purpose**: The master entry point of the FastAPI backend application.
- **Why it exists**: Every web application requires a central orchestrator where the server instance is initialized, routes are wired together, and middlewares are configured.
- **What it contains**: Creates the `FastAPI` instance, configures `CORSMiddleware` using `settings.CORS_ORIGINS`, registers the four API routers (`health_router`, `session_router`, `chat_router`, `report_router`), and defines the root `/` welcome endpoint.
- **Which other files use it**: Uvicorn server invokes it via `app.main:app`; all unit tests import it (`from app.main import app`) to initialize the `TestClient`.
- **What happens if not present**: The backend server cannot start. All API requests return connection errors.

### 2. `backend/app/config.py`
- **Purpose**: Manages global configuration parameters and environment variables.
- **Why it exists**: Hardcoding database passwords, hostnames, and ports inside source code is a major security vulnerability and makes deployment across different environments impossible.
- **What it contains**: A Pydantic `BaseSettings` class loading `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `DATABASE_URL`, and the `CORS_ORIGINS` whitelist.
- **Which other files use it**: `backend/app/main.py`, `backend/app/database.py`, and database initialization scripts.
- **What happens if not present**: Database connection strings cannot be computed, and the backend crashes immediately on startup.

### 3. `backend/app/database.py`
- **Purpose**: Initializes the SQLAlchemy database engine, session factory, and declarative ORM base.
- **Why it exists**: Establishes and manages the persistent TCP connection pool between Python and the PostgreSQL server.
- **What it contains**: `create_engine()`, `sessionmaker(autocommit=False, autoflush=False, bind=engine)`, `declarative_base()`, and the `get_db()` dependency generator function used in FastAPI route endpoints.
- **Which other files use it**: Every file in `backend/app/models/`, `backend/app/routes/`, and `backend/app/services/`.
- **What happens if not present**: No component can query or persist data to PostgreSQL.

### 4. `backend/app/routes/chat.py`
- **Purpose**: Defines the primary conversational REST endpoint `/api/chat/message`.
- **Why it exists**: Provides the HTTP gateway for survivors to submit natural language incident disclosures and follow-up question responses.
- **What it contains**: Validates `ChatMessageRequest(session_id, message)`, delegates execution to `ChatService.process_message()`, and serializes the resulting `ChatMessageResponse`.
- **Which other files use it**: Called directly by the frontend Axios client (`frontend/src/services/api.js`).
- **What happens if not present**: The chatbot interface becomes unresponsive; users cannot send messages.

### 5. `backend/app/routes/session.py`
- **Purpose**: Manages conversational session lifecycles.
- **Why it exists**: Tracks survivor state across multiple turns without requiring authentication or persistent login credentials.
- **What it contains**: `POST /api/session/start` (creates a new anonymous session and incident submission record) and `GET /api/session/{session_id}` (retrieves current session state).
- **Which other files use it**: Called when the user visits the home or chatbot pages.
- **What happens if not present**: Users cannot start new intake conversations.

### 6. `backend/app/routes/report.py`
- **Purpose**: Provides endpoints to generate and retrieve structured incident reports.
- **Why it exists**: Serves the final synthesized legal dossier to the frontend.
- **What it contains**: `POST /api/report/generate` (assembles the 10-section report via `ReportGeneratorService`) and `GET /api/report/{submission_id}` (fetches existing stored report JSON).
- **Which other files use it**: Used by `frontend/src/pages/Report.jsx` to display the final report.
- **What happens if not present**: Survivors cannot view or download their compiled legal incident reports.

### 7. `backend/app/services/chat_service.py`
- **Purpose**: The central master orchestrator for multi-turn conversational processing.
- **Why it exists**: Decouples the HTTP router from the complex business logic of coordinating deep learning models, database updates, question selection, and anti-hallucination gating.
- **What it contains**: The `ChatService.process_message()` static method, singleton loaders for `BERTIncidentPredictor` and `NEREntityPredictor`, and integration with the Dynamic Question Engine.
- **Which other files use it**: Called by `backend/app/routes/chat.py`.
- **What happens if not present**: Incoming messages cannot be processed through the ML models or question engine.

### 8. `backend/app/services/anti_hallucination.py`
- **Purpose**: Verifies that all legal provisions and helpline services returned by the system originate verbatim from verified PostgreSQL tables.
- **Why it exists**: Prevents AI hallucination from misleading survivors with non-existent legal codes or fake emergency numbers.
- **What it contains**: `validate_payload()`, `validate_legal_record()`, `validate_support_record()`, and mandatory safe fallback messages (`LEGAL_NO_MATCH_MESSAGE`).
- **Which other files use it**: Used by `ChatService` and `ReportGeneratorService`.
- **What happens if not present**: The system would lose its zero-hallucination guarantee, potentially outputting unverified claims.

### 9. `backend/app/services/report_generator.py`
- **Purpose**: Compiles all verified incident data into a structured 10-section JSON dictionary.
- **Why it exists**: Generates the formal evidentiary dossier without using generative LLMs.
- **What it contains**: `ReportGeneratorService.generate_report()` assembling Sections 1 through 10, resolving missing locations from questionnaire answers, and saving the result in the `reports` table.
- **Which other files use it**: Called by `report.py` router and `ChatService`.
- **What happens if not present**: Final reports cannot be compiled or rendered.

### 10. `backend/app/question_engine/rules.py`
- **Purpose**: Defines required information slots for each harassment category.
- **Why it exists**: Establishes the legal rules determining what facts are needed for each crime type under Indian law.
- **What it contains**: `CATEGORY_REQUIRED_ENTITIES` dictionary mapping categories (e.g., Stalking requires `PERP_REL`, `TIME_FREQ`, `PLATFORM`, `SAFETY_STATUS`), `SUPPORTED_NER_ENTITIES`, and canonical priority order.
- **Which other files use it**: Used by `engine.py` and `question_selector.py`.
- **What happens if not present**: The question engine cannot determine which questions to ask.

### 11. `ml/inference/predict_bert.py`
- **Purpose**: Production inference wrapper for the fine-tuned multi-label BERT classifier.
- **Why it exists**: Loads trained model weights, runs tokenization, computes Sigmoid probabilities, applies class-specific optimal thresholds, and returns predicted categories.
- **What it contains**: The `BERTIncidentPredictor` class, explicit evidence rule safeguards (e.g., physical domestic violence detection), and threshold tuning logic.
- **Which other files use it**: Imported by `backend/app/services/chat_service.py`.
- **What happens if not present**: Incident classification fails; the system defaults to "General Harassment".

### 12. `ml/inference/predict_ner.py`
- **Purpose**: Production inference wrapper for the fine-tuned BERT token classifier.
- **Why it exists**: Identifies entity spans (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`) in narrative text and aligns sub-word tokens with raw character spans.
- **What it contains**: The `NEREntityPredictor` class, BIO tag sequence parser (`extract_spans_from_bio`), and fast tokenizer offset mapping.
- **Which other files use it**: Imported by `backend/app/services/chat_service.py`.
- **What happens if not present**: Entity extraction fails; follow-up questions must be asked for every single slot.

### 13. `models/bert_multilabel_experiment_v2/`
- **Purpose**: Holds the production-promoted fine-tuned BERT model weights and artifacts.
- **Why it exists**: Stores the mathematical parameters resulting from fine-tuning on the 1,086-incident dataset.
- **What it contains**: `model.safetensors` (417.7 MB neural weights), `optimal_thresholds.json`, `label_mapping.json`, `training_config.json`, and tokenizer configurations.
- **Which other files use it**: Loaded by `ml/inference/predict_bert.py`.
- **What happens if not present**: Classification model cannot be loaded; PyTorch raises FileNotFoundError.

### 14. `models/ner/`
- **Purpose**: Holds the production-promoted BERT token classification model weights and artifacts.
- **Why it exists**: Stores the neural weights fine-tuned on BIO-tagged entity sequences.
- **What it contains**: `model.safetensors` (415.4 MB neural weights), `label_mapping.json` (13 BIO tags), `test_metrics.json`, and tokenizer configurations.
- **Which other files use it**: Loaded by `ml/inference/predict_ner.py`.
- **What happens if not present**: NER model cannot be loaded.

### 15. `frontend/src/services/api.js`
- **Purpose**: Configures the Axios HTTP client and defines all async API call functions.
- **Why it exists**: Centralizes network communications, resolves base URLs, and handles HTTP headers in one place.
- **What it contains**: `api` instance configured with `VITE_API_URL`, `checkHealth()`, `startSession()`, `getSession()`, `sendChatMessage()`, `generateReport()`, and `getReport()`.
- **Which other files use it**: Used by `frontend/src/context/ConversationContext.jsx`, `Chatbot.jsx`, and `Report.jsx`.
- **What happens if not present**: Frontend cannot communicate with the backend.

### 16. `frontend/src/context/ConversationContext.jsx`
- **Purpose**: Manages global React state for the entire intake journey.
- **Why it exists**: Prevents "prop drilling" and coordinates state (messages, entities, predictions, questions) across multiple pages.
- **What it contains**: `ConversationProvider`, React state hooks, and synchronization with browser `sessionStorage`.
- **Which other files use it**: Consumed by all pages via the `useConversation()` custom hook.
- **What happens if not present**: State is lost between page navigations; chatbot history resets.
'''
