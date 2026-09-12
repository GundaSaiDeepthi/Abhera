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
