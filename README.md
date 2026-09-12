# Conversational Women's Safety Legal Assistance System

## Project Overview
The **Conversational Women's Safety Legal Assistance System** is an AI-driven, privacy-focused legal aid platform. It allows users to describe safety-related incidents in natural language through an interactive chatbot interface. The system categorizes the incident using a fine-tuned multi-label BERT classifier, extracts pertinent entities via an NER model, prompts users with dynamic follow-up questions to gather missing details, and retrieves relevant legal provisions and support services from strict, authoritative datasets.

---

## Strict Knowledge & Grounding Rules
- **Legal Information Grounding**: Legal sections, titles, and explanations **MUST** originate strictly from `data/laws.csv` or corresponding PostgreSQL tables imported directly from that dataset. The system **NEVER** invents or generates fake legal advice or legal sections.
- **Support Service Grounding**: Helplines, portals, and support service details **MUST** originate strictly from `data/support_services.csv` or corresponding PostgreSQL tables imported directly from that dataset. The system **NEVER** generates fake helpline numbers or organization names.
- **Strict Fallback Requirements**:
  - If no matching legal provision exists: `"Information not available in the provided knowledge base."`
  - If no matching support service exists: `"No matching support service was found in the available support-services database."`
- **Zero-Hallucination Policy**: External LLMs are strictly forbidden from generating or inventing laws, legal advice, helpline numbers, contact details, or support organization records.

---

## Technology Stack
- **Frontend**: React.js, HTML, CSS, JavaScript (Single Page Application with Chat UI)
- **Backend**: Python, FastAPI, REST APIs, Pydantic
- **Database**: PostgreSQL (SQLAlchemy ORM)
- **Machine Learning**: Fine-tuned BERT (Multi-label Incident Classification), NER Model (Entity Extraction), PyTorch / Hugging Face Transformers
- **Testing**: pytest

---

## System Architecture

```text
               +-----------------------------------+
               |        React.js Frontend          |
               |     (Conversational Chat UI)      |
               +-----------------+-----------------+
                                 | REST APIs
                                 v
               +-----------------------------------+
               |          FastAPI Backend          |
               |                                   |
               |  +-----------------------------+  |
               |  | Incident Processing Engine  |  |
               |  +--------------+--------------+  |
               |                 |                 |
               |  +--------------v--------------+  |
               |  | ML Services (BERT / NER)    |  |
               |  +--------------+--------------+  |
               |                 |                 |
               |  +--------------v--------------+  |
               |  | Dynamic Question Engine     |  |
               |  +--------------+--------------+  |
               |                 |                 |
               |  +--------------v--------------+  |
               |  | Strict Retrieval Engine     |  |
               |  | (Laws & Support Services)   |  |
               |  +--------------+--------------+  |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------------------------+
               |        PostgreSQL Database        |
               +-----------------------------------+
```

---

## Major Modules

| Module | Location | Purpose |
| :--- | :--- | :--- |
| **Frontend** | `frontend/` | React conversational chatbot UI for user incident intake and report generation. |
| **Backend API** | `backend/` | FastAPI REST services handling sessions, messages, classification, entity extraction, law/support lookup, and dynamic question framing. |
| **Machine Learning** | `ml/` | Pipeline for fine-tuning BERT multi-label classifier and training NER entity extractors. |
| **Model Artifacts** | `models/` | Storage location for trained BERT weights, NER checkpoints, and label encoders. |
| **Database** | `database/` | PostgreSQL schema migrations and database initialization scripts. |
| **Data Directory** | `data/` | Ground-truth datasets (`laws.csv`, `support_services.csv`, `womens_safety_dataset.csv`). |
| **Documentation** | `docs/` | Architecture specs, schema diagrams, and project phase documentation. |
| **Tests** | `tests/` | Unit and integration test suites for backend APIs and ML inference pipelines. |

---

## Dataset Rules & Inventory
The system relies exclusively on authoritative datasets located in `data/`:
1. `data/laws.csv`: Definitive mappings of statutory sections (e.g., BNS, IPC, DV Act, POSH) to incident labels (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
2. `data/support_services.csv`: Verified emergency helplines, support organizations, cyber crime portals, and state/district coverage.
3. `data/womens_safety_dataset.csv`: Multi-label safety incident narratives used to fine-tune BERT and develop entity extraction.

---

## Planned Development Phases
1. **Phase 1: Project Initialization & Scaffolding** *(Current Phase)*
   - Setup project architecture, modular directory layout, configuration files, `.gitignore`, and documentation.
2. **Phase 2: Database Schema & Data Ingestion**
   - Implement PostgreSQL schemas for sessions, messages, predictions, entities, legal laws, and support services.
   - Script deterministic data ingestion from `data/laws.csv` and `data/support_services.csv`.
3. **Phase 3: Machine Learning Model Development**
   - Data preprocessing and train/validation split of `womens_safety_dataset.csv`.
   - Multi-label BERT classifier fine-tuning for categories (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
   - NER model development for entity extraction (Location, Time, Perpetrator Details, Evidence).
4. **Phase 4: Backend API & Dynamic Question Engine**
   - FastAPI REST endpoints for intake, prediction pipeline, and structured report compilation.
   - Dynamic follow-up question framing engine for unextracted/missing incident variables.
5. **Phase 5: Conversational Frontend Interface**
   - Build React.js dynamic chatbot component, message state manager, and final structured summary visualizer.
6. **Phase 6: End-to-End Integration & Testing**
   - Complete pipeline integration test, assertion of strict fallback messages, and accuracy validation.
