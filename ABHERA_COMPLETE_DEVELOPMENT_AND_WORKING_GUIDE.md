# ABHERA: Complete System Development, Engineering Architecture, and Operational Working Guide

---

## 1. Master System Overview & Project Identity

**ABHERA** — *Adaptive BERT-based Harassment Entity Recognition & Assistance* — is an enterprise-grade, neuro-symbolic artificial intelligence platform engineered to revolutionize legal literacy, trauma-informed crisis intake, and emergency support referral for victims of gender-based violence, sexual harassment, stalking, and cyber abuse.

```
+=================================================================================================+
|                                            ABHERA                                               |
|                 Adaptive BERT-based Harassment Entity Recognition & Assistance                  |
+=================================================================================================+
|  A  | Adaptive            | Dynamic follow-up questioning based on missing forensic slots       |
|  B  | BERT-based          | Fine-tuned bidirectional transformer neural encoders (110M params)  |
|  H  | Harassment          | Multi-label detection across 6 critical harassment domains          |
|  E  | Entity Recognition  | 13-tag BIO sequence labeling for forensic parameter extraction      |
|  R  | Recognition         | High-precision multi-label sequence classification (98.09% Micro-F1)|
|  A  | Assistance          | Trauma-informed statutory literacy & 3-tier local crisis aid        |
+-------------------------------------------------------------------------------------------------+
```

### The Neuro-Symbolic Hybrid Architecture
ABHERA resolves the acute safety and accuracy challenges of legal tech by cleanly decoupling **neural perception** from **symbolic knowledge retrieval**:
- **Neural Perception Layer (Deep Transformers):** Two dedicated, fine-tuned `bert-base-uncased` models ingest unstructured narrative disclosures to predict active harassment categories (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`) and extract character-level forensic entity spans (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`).
- **Symbolic Grounding Layer (Deterministic Databases):** The Dynamic Question Engine evaluates missing evidential slots via a canonical priority queue. All statutory penal provisions (Bharatiya Nyaya Sanhita 2023, IT Act 2000, POSH Act 2013, DV Act 2005) and emergency helplines are retrieved via parameterized relational queries against PostgreSQL tables, enforced by an immutable **Anti-Hallucination Service**.
- **Client-Side Document Sovereignty:** Formal legal complaint dossiers are compiled in browser memory canvas via `jsPDF`, guaranteeing that sensitive victim documents are never generated or stored on external servers.

---

## 2. Master Table of Contents & Navigation Index

This master guide serves as the unified entry point to the **15 comprehensive technical modules** located in `docs/development_guide/`. Each module provides exhaustive, production-level documentation tailored across three reading levels: **Beginner**, **Developer**, and **Technical Reviewer / Viva Examiner**.

| Module Number & File Link | Core Subject Matter & Engineering Scope | Primary Technical Deliverables |
|---|---|---|
| [Module 01: Project Origin & Requirements](file:///d:/Abhera(Mini)/docs/development_guide/01_PROJECT_ORIGIN_AND_REQUIREMENTS.md) | Problem statement, underreporting crisis, BNS 2023 legal shift, functional & non-functional requirements, system boundaries. | Neuro-symbolic design philosophy, requirement matrices, boundary definitions. |
| [Module 02: Environment & Project Setup](file:///d:/Abhera(Mini)/docs/development_guide/02_ENVIRONMENT_AND_PROJECT_SETUP.md) | Complete directory tree, technology stack, virtual environment setup, PostgreSQL configuration, verified startup commands. | Verified commands, `.env` parameter matrix, troubleshooting runbook. |
| [Module 03: Dataset & Data Preparation](file:///d:/Abhera(Mini)/docs/development_guide/03_DATASET_AND_DATA_PREPARATION.md) | 1,086 incident narratives, 6-class multi-hot vectors, 70/15/15 split, class imbalance analysis, positive weight math, leakage audit. | `prepare_classification.py` 16-field code trace, `label_mapping.json` verification. |
| [Module 04: Classical ML Baselines & TF-IDF](file:///d:/Abhera(Mini)/docs/development_guide/04_CLASSIFICATION_AND_TFIDF_BASELINES.md) | Scikit-Learn TF-IDF vectorization, OneVsRest decomposition, Logistic Regression, Linear SVM, validation threshold tuning. | Ground-truth feature selection reality check, `run_baseline_comparison.py` trace. |
| [Module 05: Deep Transformer BERT Development](file:///d:/Abhera(Mini)/docs/development_guide/05_BERT_MODEL_DEVELOPMENT.md) | BERT-base architecture, WordPiece tokenization, BCEWithLogitsLoss with pos_weight, AdamW, v1 vs v2 evolution, optimal thresholds. | 98.09% Micro-F1, 95.73% Exact Match, single false negative forensic error analysis. |
| [Module 06: Named Entity Recognition (NER)](file:///d:/Abhera(Mini)/docs/development_guide/06_NER_DEVELOPMENT.md) | BertForTokenClassification, 13-tag BIO schema, subword offset mapping, span extraction, 99.24% entity-level Micro-F1. | `predict_ner.py` 16-field code trace, character alignment mechanics. |
| [Module 07: Backend & Database Architecture](file:///d:/Abhera(Mini)/docs/development_guide/07_BACKEND_AND_DATABASE_DEVELOPMENT.md) | FastAPI asynchronous REST application, SQLAlchemy 2.0 ORM, 11 PostgreSQL relational tables, session state machine. | Master ERD diagram, table schema specifications, `ChatService` trace. |
| [Module 08: Question Engine & Conversation Flow](file:///d:/Abhera(Mini)/docs/development_guide/08_QUESTION_ENGINE_AND_CONVERSATION_FLOW.md) | Adaptive slot-filling machine, 6 forensic slots, canonical priority queue, multi-label union, anti-looping duplicate guardrails. | Dynamic question generation, state progression decision trees. |
| [Module 09: Legal, Support & Anti-Hallucination](file:///d:/Abhera(Mini)/docs/development_guide/09_LEGAL_SUPPORT_AND_ANTI_HALLUCINATION.md) | 32 BNS/IT/POSH/DV laws, 3-tier geospatial support scoring (300/200/100), Anti-Hallucination relational provenance firewall. | Verbatim database verification logic, zero-hallucination mathematical proofs. |
| [Module 10: Frontend & API Integration](file:///d:/Abhera(Mini)/docs/development_guide/10_FRONTEND_AND_API_INTEGRATION.md) | React 18 + Vite SPA, ConversationContext state synchronization, sessionStorage recovery, Axios client, live entity badging. | Component hierarchy, unidirectional dataflow, `sendChatMessage` trace. |
| [Module 11: Report & PDF Generation](file:///d:/Abhera(Mini)/docs/development_guide/11_REPORT_AND_PDF_GENERATION.md) | 10-section structured JSON report schema, client-side jsPDF compiler, ABHERA design tokens, vector checkmark rendering. | Non-generative document assembly, dynamic A4 pagination coordinate geometry. |
| [Module 12: Testing, Evaluation & Promotion](file:///d:/Abhera(Mini)/docs/development_guide/12_TESTING_EVALUATION_AND_MODEL_PROMOTION.md) | 154 automated backend unit tests (100% pass), 11-subsystem end-to-end validation, double-blind leakage audit, model promotion. | Rejection of BERT v1, formal promotion of BERT v2 to production. |
| [Module 13: Complete System Working Example](file:///d:/Abhera(Mini)/docs/development_guide/13_COMPLETE_SYSTEM_WORKING_EXAMPLE.md) | Step-by-step walkthrough of compound DV + Cyber Harassment incident across all 28 discrete execution steps. | Exact tensor transitions, logit vectors, multi-turn dialog traces. |
| [Module 14: Deployment & Maintenance](file:///d:/Abhera(Mini)/docs/development_guide/14_DEPLOYMENT_AND_MAINTENANCE.md) | Cloud hosting blueprint (Vercel frontend + Render backend + Render PostgreSQL), 2.0 GB memory sizing, security maintenance. | Production build/start commands, environment variable matrix, zero PII logging. |
| [Module 15: History, Reconstruction & Viva Guide](file:///d:/Abhera(Mini)/docs/development_guide/15_DEVELOPMENT_HISTORY_AND_VIVA_GUIDE.md) | 16-phase chronological history, 16-phase developer reconstruction guide, 12-min demonstration script, 154 viva Q&As, glossary. | Complete developer onboarding runbook, 154 technical defense answers. |

---

## 3. High-Level System Architecture Diagram

```mermaid
graph TB
    subgraph ClientLayer["Presentation Tier: React 18 + Vite SPA (Client Browser)"]
        UI[User Interface & Chatbot View]
        Context[ConversationContext - sessionStorage]
        PDF[jsPDF Vector Canvas Engine]
        Axios[Centralized Axios API Client]
        UI <--> Context
        UI --> PDF
        Context <--> Axios
    end

    subgraph APIGateway["Application Tier: FastAPI (Python 3.10 / Uvicorn)"]
        Router[REST API Routers: /session, /chat, /report, /health]
        ChatSvc[ChatService Orchestration Layer]
        QEngine[Symbolic Question Engine]
        LegalSvc[LegalMappingService]
        SuppSvc[SupportMappingService]
        AntiHalluc[AntiHallucinationService]
        RepGen[ReportGeneratorService]

        Axios <== HTTP REST / JSON ==> Router
        Router --> ChatSvc
        Router --> RepGen
        ChatSvc --> QEngine
        ChatSvc --> LegalSvc
        ChatSvc --> SuppSvc
        RepGen --> AntiHalluc
    end

    subgraph NeuralTier["Perception Tier: Pre-Loaded PyTorch Transformers"]
        BERTClassifier["BERT Multi-Label Classifier (bert_multilabel_experiment_v2)<br/>110M Params | 6 Epochs | Pos-Weighted BCE | Micro-F1: 98.09%"]
        BERTNER["BERT Named Entity Recognition (models/ner)<br/>13-Tag BIO Schema | Offset Mapping | Entity Micro-F1: 99.24%"]

        ChatSvc --> BERTClassifier
        ChatSvc --> BERTNER
    end

    subgraph StorageTier["Relational Storage Tier: PostgreSQL 16"]
        T_Users[("users")]
        T_Sessions[("sessions")]
        T_Submissions[("incident_submissions")]
        T_Messages[("conversation_messages")]
        T_Preds[("predictions")]
        T_Entities[("entities")]
        T_Questions[("questions")]
        T_Answers[("submission_question_answers")]
        T_Laws[("laws - 32 Audited BNS/IT/POSH/DV Statutes")]
        T_Support[("support_services - 12 Verified Helplines & OSCs")]
        T_Reports[("reports - Immutable 10-Section JSONs")]

        APIGateway <== SQLAlchemy 2.0 Connection Pool ==> StorageTier
    end
```

---

## 4. Key Performance Indicators & Benchmark Registry

All metrics are audited against the held-out 164-sample test partition (`ml/preprocessing/test_data.csv`) and recorded in `ml/evaluation/FINAL_EVALUATION_RESULTS.json`.

```
+=================================================================================================+
|                              ABHERA GROUND-TRUTH METRICS REGISTRY                               |
+=================================================================================================+
|  Metric Description                 | Measured Score         | Scientific Baseline Comparison   |
+-------------------------------------+------------------------+----------------------------------+
|  Total Dataset Size                 | 1,086 Samples          | 760 Train / 162 Val / 164 Test   |
|  BERT v2 Micro-F1 Score             | 98.09% (0.9809)        | vs 91.33% (BERT v1) [+6.76 pts]  |
|  BERT v2 Exact Match Accuracy       | 95.73% (0.9573)        | vs 83.54% (BERT v1) [+12.19 pts] |
|  BERT v2 Macro-F1 Score             | 97.58% (0.9758)        | vs 90.37% (BERT v1) [+7.21 pts]  |
|  BERT v2 Micro-Recall               | 99.45% (0.9945)        | 180 True Positives / 181 Cases   |
|  BERT v2 Total False Negatives      | 1 Single Miss          | vs 23 Misses (v1) [-95.65% FN]   |
|  BERT v2 Multi-Label Exact Match    | 94.12% (16 / 17 exact) | vs 0.00% (v1) [+94.12 pts]       |
|  NER Entity-Level Micro-F1          | 99.24% (0.9924)        | 100% Precision / 98.48% Recall   |
|  Backend Automated Unit Tests       | 154 Passed / 0 Failed  | 100% Pass Rate in 4.82s          |
|  End-to-End Subsystem Validation    | 11 of 11 Passed        | Certified Production Ready       |
|  Anti-Hallucination Audit Rate      | 100% Verified DB Rows  | ZERO Generative Hallucination    |
+-------------------------------------------------------------------------------------------------+
```

---

## 5. End-to-End Data Processing Flow

1. **Intake & Session Initiation:** User visits web application; a stateful `sessionStorage` session token is initialized via `POST /api/session/start`.
2. **Narrative Ingestion:** User inputs unformatted narrative text. FastAPI persists message turns to `conversation_messages` and updates `incident_submissions`.
3. **Dual Neural Feature Extraction:**
   - `BERTIncidentPredictor` evaluates `[CLS]` pooled representations across 6 harassment dimensions, applying validation-tuned optimal decision thresholds (`DV: 0.73`, `SH: 0.50`, `ST: 0.79`, `CA: 0.45`, `WH: 0.29`, `OV: 0.63`).
   - `NEREntityPredictor` executes dense token classification across the 13-tag BIO ontology, resolving character offsets for `PERP_REL`, `LOCATION`, `TIME_FREQ`, and `PLATFORM`.
4. **Adaptive Conversational Interview:**
   - `QuestionEngine` calculates set difference between category-mandated slots and known extracted information.
   - Poses prioritized follow-up questions one by one (`SAFETY_STATUS` $\succ$ `PERP_REL` $\succ$ `LOCATION` $\succ$ `TIME_FREQ` $\succ$ `PLATFORM` $\succ$ `INCIDENT_DETAILS`).
   - Stops questioning gracefully when all slots are populated or questions exhausted.
5. **Deterministic Legal & Support Assembly:**
   - `LegalMappingService` retrieves verified BNS, IT Act, POSH Act, and DV Act sections.
   - `SupportMappingService` applies 3-tier geospatial scoring: District Match (+300), State Match (+200), National Fallback (+100).
   - `AntiHallucinationService` asserts primary-key existence and verbatim string matching against PostgreSQL tables.
6. **Report Compilation & In-Browser PDF Export:**
   - `ReportGeneratorService` constructs an immutable 10-section JSON schema stored in `reports`.
   - `Report.jsx` renders visual cards and compiles an official, color-coded A4 vector PDF directly on client canvas via `jsPDF`.

---

## 6. Document Verification & Operational Sign-Off

- **Document Version:** 1.0.0
- **Audited Repository:** `D:\Abhera(Mini)`
- **Audited Git Remote:** `https://github.com/GundaSaiDeepthi/Abhera.git` (Branch: `main`)
- **Compilation Timestamp:** 2026-10-30T13:42:00Z
- **Authoritative Status:** Certified Master Development and Working Guide.
