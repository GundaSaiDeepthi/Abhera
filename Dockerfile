# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered standard I/O
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH="/app:/app/backend:${PYTHONPATH}"

# Install minimal runtime system libraries and curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU wheels first (prevents downloading 2GB+ CUDA packages)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining backend dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories (backend, ml, data, models)
COPY backend /app/backend
COPY ml /app/ml
COPY data /app/data
COPY models /app/models

# Create unprivileged application user for production container security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose backend port
EXPOSE 8000

# Container Health Check targeting FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Working directory set to backend for uvicorn discovery
WORKDIR /app/backend

# Launch FastAPI application via Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
