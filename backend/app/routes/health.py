from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """Health check endpoint confirming that the backend API service is running."""
    return {
        "status": "online",
        "message": "Conversational Women's Safety Legal Assistance System Backend is running successfully.",
        "service": "FastAPI Backend",
    }
