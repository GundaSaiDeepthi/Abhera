"""
Module for Chapter 6: Backend Development
"""

def get_chapter_06():
    return r'''# CHAPTER 6: Backend Development

---

## 6.1 What is a Backend?

In computer software architecture, a **backend** (also known as the "server-side") is the portion of a system that users cannot see directly, but which performs all the essential data processing, business logic, mathematical computations, database interactions, security checks, and machine learning inference.

While the frontend (the client) focuses on presentation, buttons, and animations on the user's screen, the backend is responsible for:
1. **Handling Business Logic**: Deciding what actions to take when data arrives.
2. **Executing Machine Learning**: Running the 110-million parameter PyTorch BERT models that require heavy CPU and RAM resources.
3. **Persisting Data**: Safely storing sessions, conversation turns, and verified legal records in PostgreSQL.
4. **Enforcing Safety & Security**: Validating incoming payloads to prevent SQL injection, and asserting anti-hallucination rules against the legal database.

---

## 6.2 Introduction to FastAPI

ABHERA uses **FastAPI**, an ultra-modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.

### Why FastAPI Was Chosen Over Flask and Django:
- **Asynchronous Execution (ASGI)**: Unlike Flask (which is synchronous and handles one request per thread by default), FastAPI is built on Starlette and runs on asynchronous event loops (`async`/`await`). This allows the server to remain responsive even during long computations.
- **Automatic Data Validation (Pydantic v2)**: Incoming request bodies are automatically parsed, type-checked, and validated against Pydantic schemas. If a client sends invalid data, FastAPI rejects it instantly with an informative `422 Unprocessable Entity` error before it ever touches database or ML code.
- **Automated OpenAPI / Swagger Documentation**: FastAPI inspects route definitions and automatically generates interactive web documentation accessible at `/docs` and `/redoc`.
- **Dependency Injection**: FastAPI features a built-in dependency injection system (`Depends()`), making it simple to manage database session lifecycles cleanly.

---

## 6.3 Application Initialization (`backend/app/main.py`)

The application entry point initializes the FastAPI app instance, configures cross-origin security, and registers modular routers:

### Actual Source Code: `backend/app/main.py`
```python
import sys
from pathlib import Path

# Ensure root project directory is in sys.path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.health import router as health_router
from app.routes.report import router as report_router
from app.routes.session import router as session_router
from app.routes.chat import router as chat_router

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Conversational Women's Safety Legal Assistance System Backend API",
)

# Configure CORS middleware for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router)
app.include_router(report_router)
app.include_router(session_router)
app.include_router(chat_router)

@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing service information and links to health check & documentation."""
    return {
        "message": "Welcome to the Conversational Women's Safety Legal Assistance System API",
        "docs_url": "/docs",
        "health_check": "/health",
    }
```

### Line-by-Line Code Explanation:
- **Lines 5-7**: Resolves the root directory path (`D:\Abhera(Mini)`) and prepends it to `sys.path`. This guarantees that Python can import the `ml` and `app` modules regardless of which subfolder Uvicorn was launched from.
- **Lines 9-15**: Imports the core `FastAPI` class, `CORSMiddleware`, application `settings`, and the four modular routers.
- **Lines 18-22**: Instantiates the `app` object with project metadata (`PROJECT_NAME = "Women's Safety Legal Assistance API"`).
- **Lines 25-31**: Adds `CORSMiddleware`. This tells the server which external web origins are permitted to make HTTP calls. `settings.CORS_ORIGINS` includes local development ports (`http://localhost:5173`, `http://localhost:3000`) and production origins (`https://abhera.vercel.app`).
- **Lines 34-37**: Mounts the four sub-routers into the application router tree.
- **Lines 40-47**: Defines a friendly root `GET /` endpoint that returns a JSON message pointing to `/docs` and `/health`.

---

## 6.4 Configuration and Environment Management (`backend/app/config.py`)

ABHERA uses Pydantic's `BaseSettings` (`pydantic-settings`) to read configuration parameters from local environment variables or an optional `.env` file:

### Actual Source Code: `backend/app/config.py`
```python
import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Women's Safety Legal Assistance API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # PostgreSQL Database Configuration
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "womens_safety_db")
    DATABASE_URL: Optional[str] = None

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5174",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

settings = Settings()
```

---

## 6.5 Database Connection & Session Management (`backend/app/database.py`)

FastAPI handles database connections using SQLAlchemy 2.0. To prevent memory leaks and thread collisions, each incoming HTTP request receives its own isolated database session that is closed automatically when the request finishes.

### Actual Source Code: `backend/app/database.py`
```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

DATABASE_URL = settings.get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # Verifies connections before giving them to workers
    pool_size=10,             # Maintains 10 persistent TCP connections
    max_overflow=20,          # Can burst up to 30 connections under heavy load
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI Dependency providing an isolated SQLAlchemy DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()            # Guarantees session is returned to pool even if exception occurs
```

---

## 6.6 The Conversational Chat Router (`backend/app/routes/chat.py`)

The `/api/chat/message` endpoint processes user messages. It demonstrates clean separation of concerns: input validation $\to$ service execution $\to$ response serialization:

### Actual Source Code: `backend/app/routes/chat.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from app.database import get_db
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["Conversational Chat"])

@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Process conversational incident message",
)
def send_chat_message(
    payload: ChatMessageRequest,
    db: DBSession = Depends(get_db),
):
    if not payload.session_id or not payload.session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty.",
        )

    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    try:
        response_data = ChatService.process_message(
            db=db,
            session_id=payload.session_id.strip(),
            user_message_text=payload.message.strip(),
        )
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}",
        )
```

---

## 6.7 Central Orchestration Service (`backend/app/services/chat_service.py`)

The `ChatService.process_message()` method is the heart of ABHERA's conversational logic:

```mermaid
flowchart TD
    Req["Incoming Message (session_id, user_message)"] --> Step1["1. Lookup / Validate Session in DB"]
    Step1 --> Step2{"Is User Answering a Follow-up Question?"}
    
    Step2 -- YES --> RecordAns["Record Answer in submission_question_answers Table"]
    Step2 -- NO --> FirstTurn["First Narrative Turn: Initialize IncidentSubmission"]
    
    RecordAns --> Merge["Merge Answer Context with Narrative Text"]
    FirstTurn --> Merge
    
    Merge --> ML_Class["Run BERT Multi-Label Classifier (predict_bert.py)"]
    Merge --> ML_NER["Run BERT Token Classifier (predict_ner.py)"]
    
    ML_Class --> EvSlots["Dynamic Question Engine: Evaluate Missing Slots"]
    ML_NER --> EvSlots
    
    EvSlots --> CheckComplete{"Are All Slots Fulfilled?"}
    
    CheckComplete -- NO --> NextQ["Select Next Priority Question (Status: QUESTIONING)"]
    CheckComplete -- YES --> Done["All Slots Complete (Status: COMPLETED)"]
    
    NextQ --> AH_Layer["Anti-Hallucination Layer: Fetch Verified Statutes & Helplines"]
    Done --> AH_Layer
    
    AH_Layer --> SaveSess["Update sessions Table in PostgreSQL"]
    SaveSess --> Resp["Return ChatMessageResponse to Client"]
```

---

## 6.8 Master REST API Reference Table

Below is the complete, verified reference table of all active REST API endpoints exposed by the ABHERA backend:

| HTTP Method | Route Endpoint | Request Payload (Input) | Success Response (Output) | Purpose & Execution Flow |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/` | None | `{"message": "...", "docs_url": "/docs", "health_check": "/health"}` | Root service welcome endpoint and documentation discovery. |
| `GET` | `/health` | None | `{"status": "online", "message": "...", "service": "FastAPI Backend"}` | Health-check endpoint confirming backend service uptime. |
| `POST` | `/api/session/start` | `{ "user_id": int (optional), "initial_narrative": str (optional) }` | `SessionResponse`: `session_id`, `submission_id`, `conversation_status: "ACTIVE"`, `created_at` | Initializes a new anonymous conversation session and creates an associated `incident_submissions` record. |
| `GET` | `/api/session/{session_id}` | Path parameter: `session_id` | `SessionResponse`: full session state, `messages`, `predicted_labels`, `entities`, `conversation_status` | Retrieves the current state of an existing session by ID; returns `404 Not Found` if session does not exist. |
| `POST` | `/api/chat/message` | `{ "session_id": str, "message": str }` | `ChatMessageResponse`: `reply_text`, `session_id`, `predicted_labels`, `entities`, `next_question`, `conversation_status` | Primary conversational intake endpoint; runs BERT classification, NER extraction, question slot evaluation, and DB legal mapping. |
| `POST` | `/api/report/generate` | `{ "submission_id": str, "narrative_text": str (opt), "state": str (opt), "district": str (opt) }` | `ReportResponse`: complete 10-section JSON report, `legal_information`, `support_services`, `audit` | Assembles and persists the formal evidentiary incident dossier; verifies all legal sections against PostgreSQL. |
| `GET` | `/api/report/{submission_id}` | Path parameter: `submission_id` | `ReportResponse`: stored 10-section JSON report data | Fetches an existing generated report from the `reports` table by submission ID; returns `404 Not Found` if missing. |
'''
