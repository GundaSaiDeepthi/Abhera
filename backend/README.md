# Backend Architecture - FastAPI REST Service

## Structure Overview
The backend application follows a clean, modular layer architecture:

- `app/api/`: REST API endpoints for session initialization, user prompt handling, follow-up answers, and report generation.
- `app/core/`: Application settings, environment configuration, constants, and fallback messages.
- `app/db/`: Database session management, PostgreSQL connection setup, and seed ingestion routines.
- `app/models/`: SQLAlchemy ORM database models mapping system entities (sessions, messages, incidents, predictions, entities, questions, reports, laws, support services).
- `app/schemas/`: Pydantic data schemas for request validation and API response formatting.
- `app/services/`: Core business logic modules including:
  - `classifier_service.py`: BERT multi-label classifier inference wrapper.
  - `ner_service.py`: Entity extraction service.
  - `question_engine.py`: Dynamic question engine identifying missing fields and generating targeted follow-up prompts.
  - `legal_retrieval_service.py`: Strict legal provision lookups directly from `data/laws.csv` or DB records.
  - `support_retrieval_service.py`: Strict support service lookups directly from `data/support_services.csv` or DB records.
  - `report_generator.py`: Final report synthesis segregating user facts, predictions, entities, legal matches, and support options.
