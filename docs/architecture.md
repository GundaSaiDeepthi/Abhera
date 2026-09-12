# System Architecture & Technical Specifications

## Architectural Design Principles
1. **Modular Architecture**: Complete separation of concerns between API routing (`backend/app/api`), business logic (`backend/app/services`), data persistence (`backend/app/models`), ML inference (`ml/`), and user UI (`frontend/`).
2. **Zero-Hallucination Legal & Support Lookup**: Statutory provisions and support emergency contacts are queried strictly against PostgreSQL database records initialized directly from `data/laws.csv` and `data/support_services.csv`.
3. **Multi-Label Incident Classification**: Incidents may encompass multiple dimensions (e.g., domestic violence + cyber abuse). The BERT model outputs independent probabilities for labels (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
4. **Structured & Segregated Reporting**: Final reports clearly partition user facts, predictions, entities, legal remedies, and support services.

## Data Flow Diagram
```text
[User Input] --> (React Chat UI)
                       |
                       v
                (FastAPI REST Endpoint)
                       |
         +-------------+-------------+
         |                           |
         v                           v
  (BERT Classifier)           (NER Extractor)
         |                           |
         +-------------+-------------+
                       |
                       v
            (Dynamic Question Engine)
                       |
                       v
      (Legal & Support Knowledge Retriever)
                       |
                       v
             (PostgreSQL Data Store)
                       |
                       v
          (Structured Report Visualizer)
```
