# ABHERA: Complete Technical Project Documentation & System Manual
**Adaptive BERT-based Harassment Entity Recognition & Assistance**

---

### Document Overview & Metadata
- **Project Name**: ABHERA (Adaptive BERT-based Harassment Entity Recognition & Assistance)
- **Domain**: Artificial Intelligence, Natural Language Processing, Machine Learning, Women's Safety & Legal Informatics
- **Jurisdictional Scope**: Republic of India (Bharatiya Nyaya Sanhita 2023, IT Act 2000, POSH Act 2013, PWDVA 2005)
- **Version**: 2.0 (Production Release — Verified & Audited)
- **Champion ML Architecture**: Fine-Tuned `bert-base-uncased` with Weighted BCE (`pos_weight`) and Class-Specific Optimal Thresholds
- **Token Classification Architecture**: Fine-Tuned `bert-base-uncased` Token Classifier with 13-Tag BIO Ontology & Sub-Word Offset Alignment
- **Knowledge Retrieval Engine**: Deterministic Relational Knowledge Base (PostgreSQL 16) with 32 Curated Legal Provisions & 12 Emergency Services
- **Automated Validation**: 154 Passing Automated Unit Tests across 11 Modules
- **Document Status**: Complete Comprehensive Master Edition (23 Chapters)

---

## Complete Table of Contents
1. [Chapter 1: Project Introduction & Abstract](#chapter-1-project-introduction--abstract)
2. [Chapter 2: Requirement Analysis](#chapter-2-requirement-analysis)
3. [Chapter 3: System Architecture & Data Flow](#chapter-3-system-architecture--data-flow)
4. [Chapter 4: Complete Project Folder Structure](#chapter-4-complete-project-folder-structure)
5. [Chapter 5: Dataset Preparation & Preprocessing](#chapter-5-dataset-preparation--preprocessing)
6. [Chapter 6: BERT Model Implementation](#chapter-6-bert-model-implementation)
7. [Chapter 7: Model Evaluation & Performance Metrics](#chapter-7-model-evaluation--performance-metrics)
8. [Chapter 8: Named Entity Recognition (NER)](#chapter-8-named-entity-recognition-ner)
9. [Chapter 9: Dynamic Follow-Up Question Engine](#chapter-9-dynamic-follow-up-question-engine)
10. [Chapter 10: Backend Implementation (FastAPI)](#chapter-10-backend-implementation-fastapi)
11. [Chapter 11: PostgreSQL Database Design](#chapter-11-postgresql-database-design)
12. [Chapter 12: Legal Mapping & Anti-Hallucination](#chapter-12-legal-mapping--anti-hallucination)
13. [Chapter 13: Frontend Implementation](#chapter-13-frontend-implementation)
14. [Chapter 14: Report Generation & Evidentiary Dossier](#chapter-14-report-generation--evidentiary-dossier)
15. [Chapter 15: Complete End-to-End Working Example](#chapter-15-complete-end-to-end-working-example)
16. [Chapter 16: Testing, Quality Assurance & Validation](#chapter-16-testing-quality-assurance--validation)
17. [Chapter 17: Complete Development Journey (16 Phases)](#chapter-17-complete-development-journey-16-phases)
18. [Chapter 18: Challenges Faced & Engineering Solutions](#chapter-18-challenges-faced--engineering-solutions)
19. [Chapter 19: Research Paper and Methodology](#chapter-19-research-paper-and-methodology)
20. [Chapter 20: Future Enhancements & Technical Roadmap](#chapter-20-future-enhancements--technical-roadmap)
21. [Chapter 21: Technical Interview Preparation (65+ Q&As)](#chapter-21-technical-interview-preparation-65-qas)
22. [Chapter 22: Project Presentation and Viva Defense](#chapter-22-project-presentation-and-viva-defense)
23. [Chapter 23: Quick Revision Cheatsheet & Summary](#chapter-23-quick-revision-cheatsheet--summary)

---

# Chapter 1: Project Introduction & Abstract

## 1.1 Project Overview & Full Form
**ABHERA** stands for **Adaptive BERT-based Harassment Entity Recognition & Assistance**. 

It is an India-focused, artificial intelligence-powered women's safety, harassment triage, and legal assistance system. The platform bridges the acute gap between experiencing an incident of harassment and receiving structured, actionable legal guidance and institutional support.

```
+-----------------------------------------------------------------------------------+
|                                  ABHERA PLATFORM                                  |
|   A - Adaptive        : Dynamic, context-aware conversational follow-up questions |
|   B - BERT-based      : Bidirectional Encoder Representations from Transformers   |
|   H - Harassment      : 6 multi-label categories (BNS, IT Act, POSH Act scope)    |
|   E - Entity          : 13-tag Named Entity Recognition (Perpetrator, Time, etc.) |
|   R - Recognition     : Sub-word offset aligned token classification              |
|   A - Assistance      : Zero-hallucination deterministic legal & support mapping  |
+-----------------------------------------------------------------------------------+
```

## 1.2 Domain and Societal Context
Harassment against women remains one of the most pervasive yet severely under-reported offenses globally and across India:
- **National Crime Records Bureau (NCRB) Data**: Crimes against women in India have shown consistent increases, with significant proportions comprising domestic cruelty (Section 498A IPC / Section 85-86 BNS), workplace harassment, cyber harassment, and physical stalking.
- **Reporting Bottlenecks**: Victims frequently hesitate to approach law enforcement due to social stigma, fear of retaliation, trauma-induced memory fragmentation, and profound confusion regarding the applicable legal frameworks.
- **The Legislative Transition (BNS 2023)**: On July 1, 2024, India enacted the Bharatiya Nyaya Sanhita (BNS 2023), replacing the Indian Penal Code (IPC 1860). Citizens and legal aid workers face immense ambiguity in identifying corresponding statutory provisions, bailability criteria, and reporting channels.
- **The LLM Hallucination Threat**: Commercial Large Language Models (e.g., ChatGPT, Gemini) frequently hallucinate non-existent statutory sections, fabricate penal code numbers, or misclassify civil remedies as criminal offenses. In legal and safety contexts, a hallucinated legal provision can invalidate an official First Information Report (FIR) or endanger a victim's safety.

## 1.3 Core Objectives of ABHERA
1. **Multi-Label Harassment Classification**: Accurately categorize unconstrained natural language incident narratives into six distinct categories:
   - Domestic Violence (`domestic_violence`)
   - Sexual Harassment (`sexual_harassment`)
   - Stalking (`stalking`)
   - Cyber Harassment (`cyber_harassment`)
   - Workplace Harassment (`workplace_harassment`)
   - Other Verbal/Physical Abuse (`other_harassment`)
2. **Fine-Grained Named Entity Recognition (NER)**: Extract vital factual anchors from the narrative using a 13-tag BIO ontology covering perpetrators, victims, relationships, temporal frequencies, platforms, and locations.
3. **Adaptive Dynamic Questioning**: Identify informational gaps required for statutory FIR drafting (such as evidence availability, frequency, or relationship) and ask targeted follow-up questions without conversational loops.
4. **Deterministic, Anti-Hallucination Legal Mapping**: Map verified harassment categories and extracted entities strictly against a curated PostgreSQL repository of 32 verified statutory sections across the Bharatiya Nyaya Sanhita (BNS 2023), Information Technology Act (IT Act 2000), POSH Act 2013, and Protection of Women from Domestic Violence Act (PWDVA 2005).
5. **Tiered Support Service Routing**: Recommend immediate assistance hotlines and support infrastructure categorized into National Helplines (112, 181, 1090), Cybercrime Portals, and State-Specific Emergency response units.
6. **Evidentiary Incident Report Dossier**: Synthesize collected narrative, classifications, entities, and legal citations into a standardized 10-section JSON dossier compiled into a tamper-evident PDF ready for legal counsel, police filing, or personal record-keeping.

## 1.4 Academic Abstract
> **Abstract** — Experiencing harassment often induces acute cognitive trauma, rendering survivors unable to formulate structured legal complaints or navigate complex penal codes. While recent advances in Large Language Models (LLMs) demonstrate impressive generative prowess, their propensity for factual hallucination poses critical hazards in legal-aid and safety-critical domains. This paper presents **ABHERA (Adaptive BERT-based Harassment Entity Recognition & Assistance)**, an end-to-end neuro-symbolic framework tailored for Indian legal jurisprudence. ABHERA couples deep bidirectional transformer models with deterministic, zero-hallucination statutory knowledge retrieval. The system implements a fine-tuned `bert-base-uncased` architecture trained with positive-class weighted Binary Cross-Entropy loss (`pos_weight`) and per-class decision threshold optimization across six distinct harassment categories, attaining a **Micro-F1 of 98.09%** and **Macro-F1 of 97.45%** on held-out test data. In parallel, a dedicated BERT token classifier extracts six entity dimensions using a 13-tag BIO schema with sub-word offset alignment, achieving a **Micro-F1 of 99.24%**. Extracted entities populate a dynamic slot-filling state machine that triggers context-sensitive, non-looping follow-up inquiries to extract critical missing evidentiary elements. Validated incidents are queried against a relational database of 32 verified statutory provisions under the Bharatiya Nyaya Sanhita (BNS 2023), Information Technology Act 2000, and POSH Act 2013, alongside a 3-tier support directory. The complete system provides an evidentiary PDF dossier generation engine validated across 154 automated unit tests, demonstrating superior domain-specific reliability compared to both classical machine learning baselines and generative language models.

---


# Chapter 2: Requirement Analysis

## 2.1 Functional Requirements (FR)
The functional requirements define the explicit behaviors, computations, and transformations executed by the ABHERA platform:

| Requirement ID | Module | Functional Specification |
| :--- | :--- | :--- |
| **FR-01** | NLP Classification | The system must accept free-form English natural language text (5 to 2000 words) and predict multi-label harassment categories with calibrated confidence scores between 0.0 and 1.0. |
| **FR-02** | Class Thresholding | The system must apply per-class optimized decision thresholds (DV: 0.73, SH: 0.50, ST: 0.79, CA: 0.45, WH: 0.29, OV: 0.63) rather than a uniform 0.50 cutoff. |
| **FR-03** | Entity Extraction | The system must extract 6 entity categories (Perpetrator, Victim, Relationship, Location, Time/Frequency, Platform) using BIO sequence labeling. |
| **FR-04** | Sub-word Alignment | The system must reconstruct complete entity character spans from WordPiece sub-word tokens via offset mapping and deduplicate overlapping spans. |
| **FR-05** | Slot Saturation | The system must track 6 conversational information slots (`relationship`, `location`, `time`, `frequency`, `platform`, `evidence`) and compute completeness status. |
| **FR-06** | Dynamic Inquiry | The system must select the highest-priority missing slot based on the primary predicted harassment category and present a targeted follow-up question. |
| **FR-07** | Anti-Looping Engine | The system must cap conversational turns at 4 rounds, prevent repeated questions for already filled slots, and terminate smoothly upon slot saturation. |
| **FR-08** | Deterministic Law | The system must query verified statutory legal provisions from PostgreSQL based on active harassment categories without any generative AI hallucination. |
| **FR-09** | Support Service Routing | The system must query verified support agencies, categorized into National (112, 181), Cyber Crime, and State-level hotlines. |
| **FR-10** | Incident Dossier | The system must compile a 10-section structured JSON incident summary encompassing session metadata, narrative, classifications, entities, answers, and legal mapping. |
| **FR-11** | PDF Compilation | The system must render a structured, printable, and downloadable PDF report on the client side using `jsPDF` without exposing private data to third parties. |
| **FR-12** | Session Persistence | The system must generate a unique UUID4 session token, persist conversational state in PostgreSQL, and maintain browser state in `sessionStorage`. |

## 2.2 Non-Functional Requirements (NFR)
The non-functional requirements define the quality attributes, performance constraints, and security standards of the system:

| Attribute | Metric / Target | Implementation in ABHERA |
| :--- | :--- | :--- |
| **Performance** | Latency < 1.5 seconds for complete inference pipeline | PyTorch CPU-optimized inference with preloaded model weights in FastAPI startup lifecycle. |
| **Reliability** | 100% deterministic legal citations | Relational database mapping using parameterized SQLAlchemy ORM queries; 0% generative hallucination. |
| **Privacy** | Zero Third-Party Telemetry | Purely self-hosted inference; incident text is never transmitted to OpenAI, Google, or external APIs. |
| **Testability** | Automated test suite coverage | 154 unit and integration tests across 11 modules covering all edge cases, schemas, and routes. |
| **Scalability** | Asynchronous I/O | FastAPI asynchronous request handlers with connection-pooled PostgreSQL backend. |
| **Usability** | Mobile-first responsive UI | React 18 with Tailwind CSS, supporting touch-friendly inputs, high contrast, and screen-reader accessibility. |
| **Fault Tolerance** | Graceful Degradation | System falls back to default thresholds and base support hotlines if specific database records are unavailable. |

## 2.3 Hardware Requirements

### Development & Training Environment
- **Processor**: Intel Core i7 / AMD Ryzen 7 (8 cores, 16 threads, 3.2 GHz+)
- **System Memory (RAM)**: 16 GB to 32 GB DDR4/DDR5
- **Storage**: 256 GB NVMe SSD (minimum 20 GB free space for checkpoints and embeddings)
- **GPU Acceleration**: NVIDIA GeForce RTX 3060 / 4060 (6 GB+ VRAM) or Google Colab T4 GPU with CUDA 11.8+

### Production / Deployment Environment (Inference Mode)
- **Processor**: 2 vCPU cores (Intel Xeon or AMD EPYC, 2.0 GHz+)
- **System Memory (RAM)**: 2 GB to 4 GB (FastAPI backend + 2 PyTorch BERT models consume ~1.4 GB resident memory)
- **Storage**: 10 GB SSD storage
- **GPU**: Not required for inference; optimized CPU tensor operations achieve < 800ms inference per request.

## 2.4 Software Requirements & Technology Stack
- **Operating System**: Windows 10/11, Ubuntu Linux 22.04 LTS, or macOS 13+
- **Programming Language**: Python 3.10.x or 3.11.x, TypeScript / JavaScript (Node.js 18+)
- **Deep Learning Framework**: PyTorch 2.0+, Hugging Face Transformers 4.35+, Hugging Face Tokenizers
- **API Framework**: FastAPI 0.104+, Uvicorn (ASGI web server), Pydantic v2
- **Database**: PostgreSQL 16.x, SQLAlchemy 2.0+ ORM, Psycopg2-binary
- **Frontend Framework**: React 18.2+, Vite 5.0+, Tailwind CSS 3.4+, Lucide React Icons
- **PDF Engine**: Client-side `jsPDF 2.5+` with `jspdf-autotable 3.8+`
- **Testing & Quality Assurance**: Pytest 7.4+, Pytest-asyncio, HTTPX

## 2.5 Target User Personas
1. **The Primary Survivor**: An individual currently experiencing harassment who requires immediate, private, trauma-informed guidance to understand their legal rights and organize evidence without intimidating legal jargon.
2. **The Legal Aid Advocate / NGO Worker**: A counselor or paralegal assisting victims who needs to rapidly transform an unstructured oral or written narrative into a legally sound incident summary citing exact BNS sections.
3. **The Law Enforcement / Cyber Cell Officer**: An investigating officer receiving an incident report accompanied by an ABHERA dossier, enabling accelerated FIR drafting with pre-extracted temporal, platform, and relational entities.

---


# Chapter 3: System Architecture & Data Flow

## 3.1 High-Level Architecture Overview
ABHERA is architected as a **Three-Tier Decoupled System** adhering to separation-of-concerns principles:
1. **Presentation Layer (Frontend)**: React 18 Single-Page Application (SPA) powered by Vite and Tailwind CSS.
2. **Application & Inference Layer (Backend API)**: Asynchronous FastAPI service orchestrating PyTorch BERT models and business logic.
3. **Persistence & Knowledge Layer (Database)**: PostgreSQL 16 relational database containing operational session state, verified statutory laws, and emergency support hotlines.

```
+---------------------------------------------------------------------------------------+
|                                   PRESENTATION TIER                                   |
|   React 18 + Vite SPA | Tailwind CSS | sessionStorage State | Client jsPDF Engine     |
+-------------------------------------------+-------------------------------------------+
                                            | JSON over REST API (HTTP/CORS)
                                            v
+---------------------------------------------------------------------------------------+
|                                    APPLICATION TIER                                   |
|   FastAPI Async Server | Uvicorn ASGI | Pydantic v2 Schemas | Dependency Injection    |
|                                                                                       |
|   +-----------------------+ +-----------------------+ +---------------------------+   |
|   |   BERT Multi-Label    | |       BERT NER        | |   Dynamic Question Engine |   |
|   |   Classification      | |   Token Classification| |   Priority Queue + State  |   |
|   |   pos_weight + Tuning | |   13-tag BIO Ontology | |   Slot Saturation Logic   |   |
|   +-----------------------+ +-----------------------+ +---------------------------+   |
|                                                                                       |
|   +-----------------------+ +-----------------------+ +---------------------------+   |
|   | Deterministic Legal   | | Tiered Support Service| | Structured Dossier        |   |
|   | Provision Matcher     | | Routing Engine        | | Generator Service         |   |
|   +-----------------------+ +-----------------------+ +---------------------------+   |
+-------------------------------------------+-------------------------------------------+
                                            | Parameterized SQL (SQLAlchemy 2.0)
                                            v
+---------------------------------------------------------------------------------------+
|                                    PERSISTENCE TIER                                   |
|   PostgreSQL 16 Relational Database                                                   |
|   - 7 Core Operational Tables (Sessions, Messages, Classes, Entities, Slots, Laws)   |
|   - 4 Audit & Analytics Tables (Metrics, Feedback, Report Exports, Audit Logs)        |
+---------------------------------------------------------------------------------------+
```

## 3.2 Data Flow Diagrams (DFD)

### DFD Level 0: System Context Diagram
The Level 0 DFD illustrates the external boundary of the ABHERA system:

```
                  +------------------------------------------------+
                  |                                                |
                  |                   USER                         |
                  |         (Survivor / Counselor)                 |
                  |                                                |
                  +-------+--------------------------------+-------+
                          |                                ^
     Natural Language     |                                |  Structured Assessment,
     Incident Narrative,  |                                |  Follow-up Questions,
     Follow-up Answers    v                                |  Legal Guidance, PDF
                  +-------+--------------------------------+-------+
                  |                                                |
                  |                    0.0                         |
                  |               ABHERA SYSTEM                    |
                  |    (Adaptive Harassment Analysis Platform)     |
                  |                                                |
                  +-------+--------------------------------^-------+
                          |                                |
     Query Parameters     |                                |  Verified Statutory
     (Categories, Scope)  v                                |  Provisions & Hotlines
                  +-------+--------------------------------+-------+
                  |                                                |
                  |              POSTGRESQL DATABASE               |
                  |         (Laws, Support Services, State)        |
                  |                                                |
                  +------------------------------------------------+
```

### DFD Level 1: Subsystem Functional Decomposition
The Level 1 DFD decomposes the system into six core operational processes:

```
[ User Input Narrative ]
         |
         v
+--------+--------+      Raw Text       +-----------------+
| 1.0 Input       |-------------------->| 2.0 BERT        |---> [ Multi-Label Probabilities ]
| Preprocessing   |                     | Classification  |---> [ Thresholded Categories ]
+--------+--------+                     +-----------------+
         |
         | Raw Text                     +-----------------+
         +----------------------------->| 3.0 BERT NER    |---> [ Extracted BIO Spans ]
                                        | Extraction      |---> [ Normalized Entity Dictionary ]
                                        +--------+--------+
                                                 |
                                                 v
+-----------------+    Inquiry Question +--------+--------+
| User Follow-Up  |<--------------------| 4.0 Dynamic     |<--- [ Extracted Entities ]
| Answers         |-------------------->| Question Engine |---> [ Saturation Check / State ]
+-----------------+                     +--------+--------+
                                                 |
                                                 v
+-----------------+    Matched Laws     +--------+--------+
| PostgreSQL      |<------------------->| 5.0 Legal &     |
| Statutory Store |    Support Services | Support Mapping |
+-----------------+                     +--------+--------+
                                                 |
                                                 v
                                        +--------+--------+
                                        | 6.0 Dossier     |---> [ 10-Section Incident JSON ]
                                        | Generation      |---> [ Client-Side PDF Report ]
                                        +-----------------+
```

### DFD Level 2: Analysis & Dynamic Questioning Subsystem
The Level 2 DFD details the stateful slot tracking and priority calculation within Process 4.0:

```
                                  [ Process 4.0 Deep Dive ]

 [ Categories & Entities ]
            |
            v
   +----------------+
   | 4.1 Slot State |=======> Read/Write [ Session Slot Store (PostgreSQL) ]
   | Initializer    |
   +--------+-------+
            |
            v
   +----------------+
   | 4.2 Priority   | <----- Primary Predicted Class (e.g., workplace_harassment)
   | Queue Sorter   | <----- Missing Slot Weights (Relationship > Platform > Evidence)
   +--------+-------+
            |
            v
   +----------------+      Yes
   | 4.3 Saturation |-------------> [ Terminate Questioning -> Trigger Report Generation ]
   | Condition Met? |
   +--------+-------+
            | No
            v
   +----------------+
   | 4.4 Template   |
   | Interpolation  |-------------> [ Emit Contextual Follow-Up Question to User ]
   +----------------+
```

## 3.3 Request-Response Lifecycles & Sequence Diagrams

### Sequence Diagram 1: Initial Analysis Request (`/api/v1/analyze`)
```
User (Browser)           FastAPI Router          BERT Classifier         BERT NER Engine        PostgreSQL DB
      |                         |                       |                       |                     |
      |-- 1. POST /analyze ---->|                       |                       |                     |
      |   {narrative: text}     |-- 2. Tokenize Text -->|                       |                     |
      |                         |-- 3. Model Forward -->|                       |                     |
      |                         |<-- 4. Sigmoids -------|                       |                     |
      |                         |-- 5. Apply Thresholds |                       |                     |
      |                         |-- 6. Tokenize Text -------------------------->|                     |
      |                         |-- 7. Model Forward -------------------------->|                     |
      |                         |<-- 8. Subword Offsets & BIO Spans ------------|                     |
      |                         |-- 9. Create Session & Persist Entities ---------------------------->|
      |                         |<-- 10. Session UUID Confirm ----------------------------------------|
      |<-- 11. 200 OK ----------|
          {session_id, categories, entities, first_question}
```

### Sequence Diagram 2: Dynamic Question Submission (`/api/v1/submit-answers`)
```
User (Browser)           FastAPI Router          Question Engine        Legal Mapper           PostgreSQL DB
      |                         |                       |                     |                       |
      |-- 1. POST /submit ----->|                       |                     |                       |
      |   {session_id, answers} |-- 2. Load Slot State ---------------------------------------------->|
      |                         |<-- 3. Current Slots ------------------------------------------------|
      |                         |-- 4. Update Slots --->|                     |                       |
      |                         |-- 5. Check Saturation>|                     |                       |
      |                         |   [Turn Count == 3    |                     |                       |
      |                         |    OR Slots Filled]   |                     |                       |
      |                         |-- 6. Query Match -------------------------->|                       |
      |                         |                                             |-- 7. Select Laws ---->|
      |                         |                                             |<-- 8. BNS Provisions -|
      |<-- 9. 200 OK -----------|
          {status: "complete", legal_provisions, support_services, summary}
```

---


# Chapter 4: Complete Project Folder Structure

## 4.1 Master Directory Tree
The ABHERA project is structured as a monorepo partitioned into backend, frontend, machine learning pipeline, documentation, and operational scripts:

```
D:\Abhera(Mini)|-- backend/                       # Python FastAPI Application & Services
|   |-- app/
|   |   |-- core/
|   |   |   |-- config.py          # Environment settings, thresholds, model paths
|   |   |   |-- database.py        # SQLAlchemy engine, sessionmaker, DB connection
|   |   |-- models/
|   |   |   |-- database_models.py # 11 SQLAlchemy ORM table definitions
|   |   |-- schemas/
|   |   |   |-- pydantic_schemas.py# Request/Response validation models (Pydantic v2)
|   |   |-- services/
|   |   |   |-- classification.py  # BERT multi-label inference & threshold logic
|   |   |   |-- ner.py             # BERT-NER inference & sub-word offset alignment
|   |   |   |-- question_engine.py # Slot state machine & dynamic priority queue
|   |   |   |-- legal_mapping.py   # Parameterized statutory legal retrieval
|   |   |   |-- support_mapping.py # Tiered emergency & helpline matching
|   |   |   |-- report.py          # 10-section JSON dossier generation
|   |   |-- api/
|   |   |   |-- v1/
|   |   |   |   |-- routes.py      # REST endpoint definitions (/analyze, /report, etc.)
|   |   |-- main.py                # FastAPI application entry point, CORS, lifespan
|   |-- tests/                     # 154 Automated Unit & Integration Tests
|   |   |-- test_classification.py # Model loading, thresholding, edge cases
|   |   |-- test_ner.py            # BIO tags, offset alignment, empty tokens
|   |   |-- test_question_engine.py# Slot priority, anti-looping, saturation
|   |   |-- test_legal_mapping.py  # BNS/IT/POSH query accuracy & anti-hallucination
|   |   |-- test_support_mapping.py# National, state, cybercell directory lookups
|   |   |-- test_report.py         # JSON dossier structural integrity
|   |   |-- test_api.py            # End-to-end HTTP route execution with TestClient
|   |   |-- test_database.py       # Session persistence & schema integrity
|   |-- venv/                      # Python virtual environment (PyTorch, Transformers)
|   |-- requirements.txt           # Python dependency manifest
|
|-- frontend/                      # React 18 + Vite Single Page Application
|   |-- src/
|   |   |-- components/
|   |   |   |-- IncidentForm.tsx   # Narrative text entry with word count & validation
|   |   |   |-- ChatInterface.tsx  # Dynamic conversational follow-up questions
|   |   |   |-- CategoryBadge.tsx  # Visual indicators for predicted harassment types
|   |   |   |-- EntityPill.tsx     # Color-coded chips for extracted NER tags
|   |   |   |-- LegalCard.tsx      # Statutory provision viewer with bailability tags
|   |   |   |-- ReportSummary.tsx  # Comprehensive dossier preview & action buttons
|   |   |-- pages/
|   |   |   |-- HomePage.tsx       # Landing page & system introduction
|   |   |   |-- AnalysisPage.tsx   # Interactive triage & dynamic question workflow
|   |   |   |-- ReportPage.tsx     # Evidentiary report review & PDF export
|   |   |   |-- AboutPage.tsx      # Methodology, ethics, and team details
|   |   |-- services/
|   |   |   |-- api.ts             # Axios / fetch client communicating with backend
|   |   |   |-- pdfExport.ts       # Client-side jsPDF + autoTable document compiler
|   |   |-- App.tsx                # React Router v6 routing & layout wrappers
|   |   |-- main.tsx               # DOM root mounting & strict mode
|   |   |-- index.css              # Tailwind CSS directives & global typography
|   |-- package.json               # Node.js dependencies (React, Vite, jsPDF, Lucide)
|   |-- vite.config.ts             # Vite bundler configuration & local dev proxy
|   |-- tailwind.config.js         # Custom color palette, typography, border radius
|
|-- ml/                            # Machine Learning Research, Training & Pipelines
|   |-- datasets/
|   |   |-- raw/
|   |   |   |-- harassment_dataset.csv   # 1,086 raw labeled incident samples
|   |   |-- processed/
|   |   |   |-- train.json / test.json   # 760 train, 162 val, 164 test split
|   |-- models/
|   |   |-- bert_classifier_v2/    # Production champion BERT multi-label weights
|   |   |   |-- pytorch_model.bin  # 438 MB fine-tuned transformer weights
|   |   |   |-- config.json        # Transformer architectural configuration
|   |   |   |-- vocab.txt          # WordPiece vocabulary (30,522 tokens)
|   |   |   |-- thresholds.json    # Verified per-class optimal cutoff values
|   |   |-- bert_ner/              # Production BERT token classification weights
|   |   |   |-- pytorch_model.bin  # 436 MB fine-tuned NER transformer weights
|   |   |   |-- config.json        # 13-tag BIO label mapping configuration
|   |-- preprocessing/
|   |   |-- prepare_classification.py    # Stratified multi-label train/val/test split
|   |   |-- audit_dataset_leakage.py     # Rigorous zero-leakage verification script
|   |-- training/
|   |   |-- train_bert_experiment_v1.py  # Unweighted baseline BERT training run
|   |   |-- train_bert_experiment_v2.py  # Champion weighted BCE + tuning pipeline
|   |   |-- train_ner.py                 # 13-tag BIO token classification training
|   |-- evaluation/
|   |   |-- run_baseline_comparison.py   # TF-IDF + Logistic Regression, SVM, RF
|   |   |-- tune_classification_thresholds.py # Grid search threshold optimizer
|   |   |-- evaluate_test_set.py         # Test set verification & confusion matrices
|
|-- docs/                          # Comprehensive Architectural & Verification Guides
|   |-- development_guide/         # 15 modular development manuals (01 to 15)
|
|-- ABHERA_COMPLETE_DOCUMENTATION/ # Master Documentation & Artifact Output Folder
|   |-- ABHERA_Complete_Project_Documentation.md
|   |-- ABHERA_Complete_Project_Documentation.pdf
|   |-- ABHERA_Project_Architecture.md
|   |-- ABHERA_API_Documentation.md
|   |-- ABHERA_Database_Documentation.md
|   |-- ABHERA_ML_and_NER_Documentation.md
|   |-- ABHERA_Interview_and_Viva_Preparation.md
```

## 4.2 Component Responsibility Matrix
The following table summarizes the key responsibility and execution boundary of each project directory:

| Directory | Layer | Tech Stack | Responsibility |
| :--- | :--- | :--- | :--- |
| `backend/app/api/v1/` | Controller | FastAPI / Pydantic | Validates incoming JSON payloads, authenticates session tokens, dispatches work to service modules. |
| `backend/app/services/` | Business Logic | PyTorch / SQLAlchemy | Executes model tensor operations, state transitions, database queries, and report serialization. |
| `backend/app/models/` | Data Modeling | SQLAlchemy 2.0 ORM | Maps relational tables (`incident_sessions`, `legal_provisions`, etc.) to Python objects. |
| `frontend/src/pages/` | Presentation | React / React Router | Manages page layouts, URL navigation, and top-level user experience flows. |
| `frontend/src/services/` | Client Utilities | TypeScript / jsPDF | Handles HTTP communication with backend and runs client-side vector PDF generation. |
| `ml/training/` | Offline ML Pipeline | PyTorch / Transformers | Implements supervised fine-tuning loops, checkpointing, and hyperparameter scheduling. |
| `ml/evaluation/` | Offline Validation | Scikit-learn / NumPy | Computes confusion matrices, F1 scores, ROC-AUC curves, and decision thresholds. |

---


# Chapter 5: Dataset Preparation & Preprocessing

## 5.1 Dataset Acquisition and Provenance
The ABHERA dataset comprises **1,086 carefully curated, unconstrained incident descriptions** reflecting real-world harassment scenarios within the Indian socio-legal context:
- **Corpus Sources**: Curated from public legal dockets, appellate court judgments (Supreme Court of India, High Courts), legal aid case summaries, anonymized grievance portals, and synthetic seed expansions vetted by legal domain experts.
- **Linguistic Profile**: English incident narratives exhibiting authentic conversational variance, colloquial Indian English expressions, legal and non-legal vernacular, spelling irregularities, and emotional distress markers.
- **Multi-Label Characteristic**: Unlike artificial single-label benchmarks, real harassment incidents frequently span multiple overlapping legal violations. For instance, an incident where an estranged husband repeatedly sends abusive messages on WhatsApp and threatens the victim at her office constitutes Domestic Violence, Stalking, Cyber Harassment, and Workplace Harassment simultaneously.

## 5.2 Multi-Label Class Taxonomy
The dataset annotates each narrative across six operational harassment categories:

| Harassment Category | Code | Description & Legal Context | Total Samples | % of Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Domestic Violence** | `domestic_violence` | Physical, emotional, sexual, or economic abuse by an intimate partner, spouse, or domestic relative (BNS Sec 85-86, PWDVA 2005). | 264 | 24.31% |
| **Sexual Harassment** | `sexual_harassment` | Unwelcome physical contact, sexual advances, sexually colored remarks, or requests for sexual favors (BNS Sec 75). | 252 | 23.20% |
| **Stalking** | `stalking` | Repeated following, physical monitoring, or persistent communication attempts despite clear disinterest (BNS Sec 78). | 240 | 22.09% |
| **Cyber Harassment** | `cyber_harassment` | Online abuse, non-consensual image sharing, cyberstalking, trolling, or impersonation via digital platforms (IT Act Sec 66E/67/67A). | 236 | 21.73% |
| **Workplace Harassment** | `workplace_harassment` | Quid pro quo harassment, hostile work environments, or abuse within organizational hierarchies (POSH Act 2013). | 198 | 18.23% |
| **Other Harassment** | `other_harassment` | Public verbal abuse, street harassment (eve-teasing), criminal intimidation, or insults to modesty (BNS Sec 79, 351). | 172 | 15.84% |

*Note: Total label instances across the 1,086 samples equal 1,362, confirming a label density of 1.254 labels per incident narrative.*

## 5.3 Text Preprocessing Pipeline
Raw natural language narratives undergo disciplined preprocessing implemented in `ml/preprocessing/prepare_classification.py`:
1. **Unicode Normalization**: Conversion of smart quotes, em-dashes, and exotic UTF-8 characters to standard ASCII equivalents.
2. **Whitespace Stripping & Canonicalization**: Elimination of excessive tab characters, multiple spaces, and carriage returns.
3. **Preservation Strategy**: Crucially, punctuation (question marks, exclamation marks), casing, and emotional indicators are **NOT stripped**. BERT's WordPiece tokenizer and self-attention mechanisms leverage punctuation and capitalization to infer sentiment, urgency, and sentence boundary syntax.

## 5.4 Stratified Multi-Label Dataset Splitting
Partitioning multi-label datasets requires specialized iterative stratification to preserve label distribution across splits without sample contamination:

```
+-----------------------------------------------------------------------------+
|                          COMPLETE DATASET (1,086 Samples)                   |
+-----------------------------------------------------------------------------+
         |                                |                               |
         v (70.0%)                        v (14.9%)                       v (15.1%)
+--------------------+           +--------------------+          +--------------------+
|   TRAINING SET     |           |  VALIDATION SET    |          |    TEST SET        |
|    760 Samples     |           |    162 Samples     |          |   164 Samples      |
| Used for gradient  |           | Used for early     |          | Held-out gold set  |
| weight updates     |           | stopping & tuning  |          | for evaluation     |
+--------------------+           +--------------------+          +--------------------+
```

## 5.5 Rigorous Data Leakage Audit
To eliminate methodological flaws common in academic NLP projects, an automated audit script (`ml/preprocessing/audit_dataset_leakage.py`) was executed against the partitions:
- **Direct Duplicate Audit**: 0 overlapping identical text samples between Train, Val, and Test ($Leakage = 0.0\%$).
- **Normalized Fuzzy Overlap**: 0 samples with Levenshtein similarity $> 0.85$ across split boundaries.
- **Entity Leakage Check**: Confirmed that test set evaluation evaluates generalization across unseen syntactic formulations rather than memorized identity strings.

## 5.6 Positive Class Weight (`pos_weight`) Calculation
Because negative instances outnumber positive instances for any given label (average positive prevalence ~20%), standard binary cross-entropy induces majority-class bias. To compensate, we calculate the exact positive class weight vector for PyTorch's `BCEWithLogitsLoss`:

$$	ext{pos\_weight}_c = rac{N - P_c}{P_c}$$

Where $N = 760$ (training samples), and $P_c$ is the positive count for category $c$ in the training set:
- **Domestic Violence**: $P_{DV} = 185 \implies 	ext{pos\_weight} = (760 - 185) / 185 = 3.108$
- **Sexual Harassment**: $P_{SH} = 176 \implies 	ext{pos\_weight} = (760 - 176) / 176 = 3.318$
- **Stalking**: $P_{ST} = 168 \implies 	ext{pos\_weight} = (760 - 168) / 168 = 3.524$
- **Cyber Harassment**: $P_{CA} = 165 \implies 	ext{pos\_weight} = (760 - 165) / 165 = 3.606$
- **Workplace Harassment**: $P_{WH} = 139 \implies 	ext{pos\_weight} = (760 - 139) / 139 = 4.468$
- **Other Harassment**: $P_{OV} = 121 \implies 	ext{pos\_weight} = (760 - 121) / 121 = 5.281$

---


# Chapter 6: BERT Model Implementation

## 6.1 Architectural Selection: Why `bert-base-uncased`?
ABHERA adopts the pre-trained `bert-base-uncased` transformer architecture (Devlin et al., 2018):
- **Bidirectional Self-Attention**: Unlike autoregressive models (e.g., GPT) which read left-to-right, BERT's multi-head bidirectional self-attention simultaneously conditions on both preceding and succeeding context tokens. This is critical for understanding conversational harassment narratives where the significance of an action depends on later disclosures.
- **Model Parameters**: 12 transformer encoder layers ($L=12$), 768 hidden dimension size ($H=768$), 12 self-attention heads ($A=12$), totaling **110 million parameters**.
- **WordPiece Tokenizer**: Vocabulary of 30,522 tokens capable of decomposing out-of-vocabulary and misspelled terms into meaningful sub-words (e.g., `harassing` $	o$ `harass` + `##ing`).

```
Input Tokens: [CLS]  My  ex-  husband  threatened  me  on  WhatsApp  [SEP]
                |     |    |      |         |       |   |      |        |
         +--------------------------------------------------------------+
         |                 12-Layer Transformer Encoder                 |
         |                   (Bidirectional Attention)                  |
         +--------------------------------------------------------------+
                |     |    |      |         |       |   |      |        |
Hidden Reps:   h_CLS  h1   h2     h3        h4      h5  h6     h7      h_SEP
                |
                v
         +--------------+
         | Dropout(0.2) |
         +--------------+
                |
                v
         +-----------------------------+
         | Linear Classifier (768 -> 6)|
         +-----------------------------+
                |
                v
         Logits [z_1, z_2, z_3, z_4, z_5, z_6]
                |
                v Sigmoid Function
         Probabilities [p_1, p_2, p_3, p_4, p_5, p_6] in [0, 1]
```

## 6.2 Mathematical Formulation of Multi-Label Classification
Multi-label classification assigns an independent binary decision to each of the $K=6$ categories. The model outputs a raw logit vector $\mathbf{z} \in \mathbb{R}^6$ from the pooled `[CLS]` embedding:

$$p_c = \sigma(z_c) = rac{1}{1 + e^{-z_c}} \quad orall c \in \{1, 2, \dots, 6\}$$

The loss is computed via Weighted Binary Cross-Entropy (`BCEWithLogitsLoss`):

$$\mathcal{L} = -rac{1}{K} \sum_{c=1}^K \left[ w_c \cdot y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) ight]$$

Where:
- $y_c \in \{0, 1\}$ is the ground truth indicator.
- $w_c = 	ext{pos\_weight}_c$ penalizes false negatives on minority classes.
- Log-sum-exp stabilization prevents floating-point overflow during backpropagation.

## 6.3 Optimization & Hyperparameter Configuration
Training was executed using the AdamW optimizer with decoupled weight decay:
- **Batch Size**: 16
- **Peak Learning Rate**: $2 	imes 10^{-5}$
- **Learning Rate Scheduler**: Linear warmup for first 10% of total training steps, followed by linear decay to 0.0.
- **Weight Decay**: 0.01 (applied to non-bias and non-LayerNorm parameters).
- **Epochs**: 5 epochs with checkpoint evaluation after every epoch.
- **Maximum Sequence Length**: 256 tokens (covers 99.4% of incident narratives without truncation).

## 6.4 Evolution: BERT v1 vs. BERT v2
During development, two distinct iterations of the classification model were engineered:

| Characteristic | BERT v1 (Baseline Exploration) | BERT v2 (Production Champion) | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Loss Function** | Standard Unweighted `BCELoss` | `BCEWithLogitsLoss` with `pos_weight` | Compensates for 80% negative class prevalence per label. |
| **Decision Cutoff** | Uniform 0.50 threshold for all classes | Class-specific optimized thresholds | Acknowledges differing risk profiles and confidence distributions. |
| **Micro-F1 Score** | 94.62% | **98.09%** | $+3.47\%$ overall performance gain. |
| **False Negatives** | 9 missed violations | **1 missed violation (TEST_0050)** | $88.9\%$ reduction in hazardous false negative misclassifications. |
| **Validation Loss** | 0.142 | **0.068** | Superior model convergence and calibration. |

## 6.5 Per-Class Decision Threshold Optimization
In safety-critical triage, using a naive 0.50 cutoff fails because some classes (e.g., workplace harassment) express subtle contextual indicators requiring higher sensitivity, while other classes (e.g., stalking) require strict specificity.
Using grid search optimization on the validation set (`ml/evaluation/tune_classification_thresholds.py`), the optimal thresholds were derived:

| Class Code | Category Name | Optimal Threshold | Operational Behavior |
| :--- | :--- | :--- | :--- |
| `domestic_violence` | Domestic Violence | **0.73** | High threshold eliminates false alarms from non-domestic altercations. |
| `sexual_harassment` | Sexual Harassment | **0.50** | Balanced sensitivity and specificity for overt physical/verbal offenses. |
| `stalking` | Stalking | **0.79** | High precision prevents misclassifying isolated inquiries as stalking. |
| `cyber_harassment` | Cyber Harassment | **0.45** | Moderate threshold captures diverse online harassment modalities. |
| `workplace_harassment` | Workplace Harassment | **0.29** | Highly sensitive threshold captures veiled professional retaliations. |
| `other_harassment` | Other Abuse | **0.63** | Conservative cutoff reserved for explicit street abuse and insults. |

---


# Chapter 7: Model Evaluation & Performance Metrics

## 7.1 Mathematical Foundations of Evaluation Metrics
In multi-label classification, evaluation must be computed globally across all decisions (Micro-averaging) and across each label independently (Macro-averaging):

$$	ext{Precision} = rac{	ext{TP}}{	ext{TP} + 	ext{FP}}, \quad 	ext{Recall} = rac{	ext{TP}}{	ext{TP} + 	ext{FN}}$$

$$	ext{F1-Score} = 2 	imes rac{	ext{Precision} 	imes 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$$

$$	ext{Micro-F1} = rac{2 \sum_{c=1}^K 	ext{TP}_c}{2 \sum_{c=1}^K 	ext{TP}_c + \sum_{c=1}^K 	ext{FP}_c + \sum_{c=1}^K 	ext{FN}_c}$$

$$	ext{Macro-F1} = rac{1}{K} \sum_{c=1}^K 	ext{F1}_c$$

## 7.2 Comprehensive Test Set Evaluation Results
Evaluation on the held-out gold test set ($N=164$ samples $	imes 6$ classes $= 984$ binary classification decisions) yielded the following definitive metrics:

```
+-----------------------------------------------------------------------------------+
|                        CONFUSION MATRIX (164 TEST SAMPLES)                        |
|                                                                                   |
|                           Ground Truth Positive    Ground Truth Negative          |
|   Predicted Positive        TP = 180                 FP = 6                       |
|   Predicted Negative        FN = 1                   TN = 797                     |
|                                                                                   |
|   Total Decisions = 984 | Actual Positives = 181 | Actual Negatives = 803         |
+-----------------------------------------------------------------------------------+
```

### Detailed Metric Computations
- **Micro-Precision**: $rac{180}{180 + 6} = rac{180}{186} = \mathbf{96.77\%}$
- **Micro-Recall**: $rac{180}{180 + 1} = rac{180}{181} = \mathbf{99.45\%}$
- **Micro-F1**: $2 	imes rac{0.9677 	imes 0.9945}{0.9677 + 0.9945} = \mathbf{98.09\%}$
- **Macro-F1**: $\mathbf{97.45\%}$
- **Subset Accuracy (Exact Match Ratio)**: $\mathbf{95.73\%}$ (157 of 164 samples had 100% exact predicted label vectors).

## 7.3 Per-Class Performance Breakdown
The following table presents the verified performance of the champion BERT v2 model across each individual harassment category:

| Category | Optimal Threshold | Support (Positives) | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Domestic Violence** | 0.73 | 36 | 100.00% | 100.00% | **100.00%** | Perfect |
| **Sexual Harassment** | 0.50 | 34 | 97.14% | 100.00% | **98.55%** | Excellent |
| **Stalking** | 0.79 | 33 | 97.06% | 100.00% | **98.51%** | Excellent |
| **Cyber Harassment** | 0.45 | 32 | 96.97% | 100.00% | **98.46%** | Excellent |
| **Workplace Harassment** | 0.29 | 26 | 96.15% | 96.15% | **96.15%** | 1 FN |
| **Other Harassment** | 0.63 | 20 | 90.91% | 100.00% | **95.24%** | 2 FP |
| **Overall Micro** | — | **181** | **96.77%** | **99.45%** | **98.09%** | SOTA |
| **Overall Macro** | — | **181** | **96.37%** | **99.36%** | **97.45%** | SOTA |

## 7.4 Comparative Baseline Benchmark
To substantiate the technical necessity of deep transformer models, classical machine learning baselines were evaluated on the identical stratified test set (`ml/evaluation/run_baseline_comparison.py`):

| Model Architecture | Feature Representation | Precision (Micro) | Recall (Micro) | Micro-F1 | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF (3,000 n-grams) | 89.21% | 87.65% | 88.42% | 87.12% |
| **Random Forest** | TF-IDF (3,000 n-grams) | 91.03% | 78.29% | 84.18% | 82.90% |
| **Linear SVM** | TF-IDF (3,000 n-grams) | 90.45% | 89.08% | 89.76% | 88.64% |
| **BERT v1 (Standard)** | Dense WordPiece [CLS] | 94.15% | 95.10% | 94.62% | 93.85% |
| **BERT v2 (ABHERA)** | Dense [CLS] + pos_weight | **96.77%** | **99.45%** | **98.09%** | **97.45%** |

### Why Classical Baselines Fail
1. **Bag-of-Words Limitation**: TF-IDF models cannot understand syntactic negation or word order. In "My boss asked me to stay late for the project," classical models flag harassment due to "boss" and "late", whereas BERT recognizes benign context.
2. **Polysemy & Slang**: Terms like "block" (blocking on Instagram vs blocking a doorway) require context-dependent token representations.
3. **Sparse Representation**: TF-IDF vectors of sparse n-grams fail on diverse colloquial expressions not present in training n-gram dictionaries.

## 7.5 Forensic Error Analysis: The Single False Negative (`TEST_0050`)
Rigorous scientific integrity requires full transparency regarding model limitations:
- **Sample Identifier**: `TEST_0050`
- **Narrative**: *"During my annual performance review, my team lead remarked that promotions in our department require 'flexibility outside office hours' and later reassigned my core client accounts to a junior colleague after I declined his dinner invitation."*
- **Ground Truth**: `workplace_harassment = 1`, `sexual_harassment = 1`
- **Model Predictions**:
  - `sexual_harassment`: Probability $0.78 > 0.50 \implies \mathbf{1}$ (**Correct TP**)
  - `workplace_harassment`: Probability $0.24 < 0.29 \implies \mathbf{0}$ (**False Negative FN**)
- **Root Cause Analysis**: The phrase "annual performance review" and "client accounts" carried business vocabulary that dampened the workplace harassment logits just below the 0.29 threshold. However, because `sexual_harassment` was correctly detected with 78% confidence, the downstream legal mapping still successfully triggered BNS Section 75 and POSH Act provisions.

---


# Chapter 8: Named Entity Recognition (NER)

## 8.1 NER Task Formulation & Societal Necessity
While classification identifies *what categories of harassment* occurred, legal complaints (such as Police First Information Reports) strictly require factual particulars:
- *Who committed the act?*
- *What is their relationship to the complainant?*
- *Where did it occur?*
- *When and how frequently did it happen?*
- *Which digital platform or medium was utilized?*

ABHERA implements a dedicated token classification model (`bert-base-uncased`) fine-tuned to extract these critical evidentiary anchors.

## 8.2 The 13-Tag BIO Taxonomy
The model classifies every sub-word token into one of 13 BIO (Beginning, Inside, Outside) tags across 6 operational entity categories:

| Entity Category | BIO Tags | Real-World Examples | Evidentiary Purpose |
| :--- | :--- | :--- | :--- |
| **Perpetrator** | `B-PERP`, `I-PERP` | "Ramesh Sharma", "my senior manager", "the cab driver" | Identifies accused party for investigation. |
| **Victim** | `B-VICTIM`, `I-VICTIM` | "me", "my sister", "Pooja", "the complainant" | Identifies affected person and standing. |
| **Relationship** | `B-PERP_REL`, `I-PERP_REL` | "husband", "colleague", "stranger", "landlord" | Determines applicability of PWDVA or POSH Act. |
| **Location** | `B-LOCATION`, `I-LOCATION` | "office cafeteria", "Metro station", "matrimonial home" | Establishes jurisdictional police station limits. |
| **Time & Frequency** | `B-TIME_FREQ`, `I-TIME_FREQ` | "every night", "last Friday at 9 PM", "for 6 months" | Establishes limitation periods and ongoing course of conduct. |
| **Platform** | `B-PLATFORM`, `I-PLATFORM` | "WhatsApp", "Instagram DM", "official email", "phone call" | Directs digital forensics and IT Act Section 67 invocation. |
| **Outside** | `O` | Any non-entity grammatical word | Background tokens. |

## 8.3 Sub-Word Token Alignment & Offset Mapping
A major challenge when using BERT for Named Entity Recognition is **WordPiece sub-word tokenization**:
- Words are frequently split into multiple sub-tokens (e.g., `WhatsApp` $	o$ `['whats', '##ap', '##p']`).
- If only the first sub-token receives `B-PLATFORM` and subsequent tokens receive `O`, standard string reconstruction produces corrupted text fragments.

### The Alignment Algorithm in `backend/app/services/ner.py`
1. During tokenization, retrieve `return_offsets_mapping=True` from the tokenizer, providing character spans `(start_char, end_char)` for each token.
2. Assign the entity label of the parent word to the initial sub-token, and either propagate `I-TAG` or mask out subsequent sub-tokens with index `-100` during loss computation so they do not distort cross-entropy gradients.
3. During inference, reconstruct the continuous string span using character offsets directly from the original raw input string, preserving original capitalization and punctuation.
4. Merge consecutive or adjacent tokens of the same entity category and filter out single-character punctuation artifacts.

```
Original Narrative: "My husband beat me at home yesterday"
Offsets:             (0,2) (3,10)   (11,15) (16,18) (19,21) (22,26) (27,36)
Tokens:             My    husband  beat    me      at      home    yesterday
Ground Truth BIO:   O     B-REL    O       B-VIC   O       B-LOC   B-TIME
Predicted Entities: {
  "relationship": ["husband"],
  "victim": ["me"],
  "location": ["home"],
  "time": ["yesterday"]
}
```

## 8.4 NER Performance & Per-Label Recall Analysis
The fine-tuned BERT-NER model was evaluated on the independent test split, achieving exceptional token-level and entity-level performance:

```
+-----------------------------------------------------------------------------+
|                         BERT-NER EVALUATION SUMMARY                         |
|   Overall Token Micro-F1 : 99.24%                                           |
|   Overall Precision      : 98.92%                                           |
|   Overall Recall         : 99.56%                                           |
+-----------------------------------------------------------------------------+
```

### Per-Entity Recall & Precision Table
In safety and legal triage, **Recall** is the most critical metric: missing an entity (e.g., failing to extract that the perpetrator was the victim's "husband") prevents the system from triggering domestic violence protections.

| Entity Dimension | Precision | Recall | F1-Score | Extraction Reliability |
| :--- | :--- | :--- | :--- | :--- |
| **Relationship (`PERP_REL`)** | 99.12% | **100.00%** | **99.56%** | Perfect recall: never misses domestic/workplace ties. |
| **Time & Frequency (`TIME_FREQ`)** | 98.85% | **100.00%** | **99.42%** | Perfect recall: captures continuous harassment patterns. |
| **Platform (`PLATFORM`)** | 99.40% | **100.00%** | **99.70%** | Perfect recall: captures digital messaging apps flawlessly. |
| **Perpetrator (`PERP`)** | 98.75% | **99.10%** | **98.92%** | High precision and recall across names and roles. |
| **Victim (`VICTIM`)** | 98.10% | **98.50%** | **98.30%** | Consistently identifies self-referential pronouns & names. |
| **Location (`LOCATION`)** | 95.80% | **92.31%** | **94.02%** | Minor drop on ambiguous indoor locations (e.g., "stairwell"). |

---


# Chapter 9: Dynamic Follow-Up Question Engine

## 9.1 Motivation: Bridging Information Gaps
When a person experiences trauma, their initial incident disclosure is often fragmented, highlighting immediate distress while omitting vital legal particulars required for police reporting or statutory relief. For instance, a narrative like *"A guy is harassing me every day"* lacks:
- The relationship to the perpetrator (Determines if BNS Sec 85 Domestic Violence or POSH Act applies)
- The platform or medium used (Determines if IT Act Sec 67 applies)
- The existence of physical or digital evidence (Determines bailability and evidentiary strength)

Standard chatbot systems either interrogate the user with rigid 20-question forms or drift into unconstrained conversations. ABHERA implements an **Adaptive Slot-Filling State Machine** that dynamically ranks informational gaps and terminates smoothly.

## 9.2 The Six Information Slots
The engine maintains and updates state across six operational slots:

```
+-----------------------------------------------------------------------------------+
|                         INFORMATION SLOTS MONITORED                               |
|   1. relationship : Connection between victim and perpetrator (stranger, spouse)  |
|   2. location     : Physical address, office, public transit, or residential area |
|   3. time         : Date, time of day, or temporal anchor of occurrences          |
|   4. frequency    : One-time incident vs continuous pattern (repeated, daily)     |
|   5. platform     : In-person vs digital vector (WhatsApp, Instagram, phone)      |
|   6. evidence     : Availability of artifacts (screenshots, CCTV, audio, witness) |
+-----------------------------------------------------------------------------------+
```

## 9.3 Dynamic Priority Queue Algorithm
Rather than querying missing slots at random, the engine dynamically prioritizes questions based on the **primary predicted harassment category** identified by the BERT classifier:

```
Algorithm: Dynamic Slot Selection
Input: 
  - Primary Class C_top
  - Currently Populated Slots S_filled
  - Conversational Turn Count T
Output:
  - Next Follow-Up Question Q_next OR Termination Signal

1. If T >= 3 OR all slots in S_filled are populated:
     Return TERMINATE (Proceed to Legal Mapping & Report Generation)

2. Define Class-Specific Priority Hierarchies:
     If C_top == "workplace_harassment":
        PriorityOrder = [relationship, platform, evidence, frequency, location, time]
     Else If C_top == "domestic_violence":
        PriorityOrder = [relationship, frequency, evidence, time, location, platform]
     Else If C_top == "cyber_harassment":
        PriorityOrder = [platform, evidence, relationship, frequency, time, location]
     Else If C_top == "stalking":
        PriorityOrder = [frequency, platform, location, relationship, evidence, time]
     Else:
        PriorityOrder = [relationship, location, time, frequency, platform, evidence]

3. For each slot S in PriorityOrder:
     If S NOT in S_filled:
        SelectedSlot = S
        Break

4. Retrieve Question Template for SelectedSlot conditioned on C_top.
5. Return Q_next
```

## 9.4 Context-Aware Question Template Catalog
To maintain a supportive, trauma-informed conversational tone, questions are dynamically formulated using empathetic phrasing:

| Target Slot | Primary Harassment Context | Adaptive Follow-Up Question Template |
| :--- | :--- | :--- |
| `relationship` | Domestic Violence / General | *"To help determine applicable protections, could you share how you know this person (e.g., spouse, family member, acquaintance, or stranger)?"* |
| `platform` | Cyber Harassment | *"Which digital platform or application did this occur on (such as WhatsApp, Instagram, email, or direct phone call)?"* |
| `frequency` | Stalking | *"How often has this behavior occurred (e.g., was it an isolated incident, or has it been happening repeatedly over days or weeks)?"* |
| `evidence` | Any Category | *"Do you currently have any saved records, such as messages, call logs, emails, screenshots, or witnesses who observed the incident?"* |
| `location` | Workplace / In-Person | *"Where did this incident primarily take place (for example, inside an office building, university campus, public transit, or at home)?"* |

## 9.5 Anti-Looping and Graceful Termination Safeguards
In stateful conversational agents, infinite inquiry loops and repetitive questions cause user frustration. ABHERA implements three deterministic safeguards:
1. **Turn Cap Enforcement**: Strict cap of maximum 3 to 4 turns ($T_{\max} = 3$). Once reached, questioning immediately terminates regardless of unfilled slots.
2. **Strict Non-Repetition Filter**: Any slot populated during the initial NER pass or answered in an earlier turn is permanently marked `FILLED` and excluded from candidate queues.
3. **User Skip Handling**: If the user responds with non-committal phrases (*"I don't know"*, *"skip"*, *"prefer not to say"*), the slot is flagged as `SKIPPED` rather than re-queried.

---


# Chapter 10: Backend Implementation (FastAPI)

## 10.1 Framework Selection: Why FastAPI?
ABHERA's backend application is implemented using **FastAPI 0.104+** running on top of **Uvicorn** (ASGI):
- **Asynchronous Concurrency**: Built-in Python `async`/`await` support allows high-throughput concurrent request handling without blocking the server thread while awaiting database queries or tensor computations.
- **Pydantic v2 Serialization**: Rigorous runtime data validation with automatic generation of OpenAPI 3.0 (Swagger) schemas.
- **Lifespan Context Management**: Neural network model weights (BERT classifier and BERT-NER) are loaded once during server startup into application memory, eliminating per-request loading latency.

## 10.2 REST API Routes Catalog
The backend exposes clean, versioned REST endpoints under `/api/v1/`:

| Method | Endpoint | Description | Request Body | Response Payload |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/analyze` | Initiates incident triage: classifies text, extracts entities, opens session. | `{ "narrative": string }` | `{ "session_id": UUID, "categories": list, "entities": dict, "first_question": string }` |
| `POST` | `/api/v1/follow-up` | Requests next dynamic question based on current session state. | `{ "session_id": UUID }` | `{ "session_id": UUID, "question": string, "slot": string, "completed": bool }` |
| `POST` | `/api/v1/submit-answers` | Submits answers to follow-up questions, updates slot state. | `{ "session_id": UUID, "answers": dict }` | `{ "session_id": UUID, "slots": dict, "next_question": string, "completed": bool }` |
| `POST` | `/api/v1/report` | Compiles full 10-section structured JSON incident dossier. | `{ "session_id": UUID }` | `{ "session_id": UUID, "dossier": object, "timestamp": string }` |
| `GET` | `/api/v1/legal-provisions` | Queries verified legal provisions catalog with optional category filter. | Query param: `?category=...` | `{ "total": int, "provisions": list }` |
| `GET` | `/api/v1/support-services` | Queries emergency hotlines with optional level or category filter. | Query param: `?level=national` | `{ "total": int, "services": list }` |
| `GET` | `/health` | Liveness and readiness probe for uptime monitoring. | None | `{ "status": "healthy", "models_loaded": true, "database": "connected" }` |

## 10.3 Pydantic v2 Schema Contracts
Strict schema enforcement ensures type safety and prevents malformed data ingestion (`backend/app/schemas/pydantic_schemas.py`):

```python
from pydantic import BaseModel, Field, UUID4
from typing import List, Dict, Optional

class IncidentSubmissionRequest(BaseModel):
    narrative: str = Field(..., min_length=15, max_length=10000, 
                           description="Natural language incident description in English")

class HarassmentCategoryPrediction(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_positive: bool
    threshold_applied: float

class EntityExtractionResponse(BaseModel):
    relationship: List[str] = []
    location: List[str] = []
    time: List[str] = []
    frequency: List[str] = []
    platform: List[str] = []
    evidence: List[str] = []

class IncidentAnalysisResponse(BaseModel):
    session_id: UUID4
    categories: List[HarassmentCategoryPrediction]
    entities: EntityExtractionResponse
    next_question: Optional[str] = None
    target_slot: Optional[str] = None
    completed: bool = False
```

## 10.4 Lifespan State Management & Model Caching
To maintain sub-second response times, models are preloaded during server initialization (`backend/app/main.py`):

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.classification import load_classifier
from app.services.ner import load_ner_model

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load transformer models into memory
    print("Loading BERT Classification and NER weights...")
    ml_models["classifier"] = load_classifier()
    ml_models["ner"] = load_ner_model()
    yield
    # Shutdown: Clean up GPU/RAM resources
    ml_models.clear()

app = FastAPI(title="ABHERA API", version="1.0.0", lifespan=lifespan)
```

---


# Chapter 11: PostgreSQL Database Design

## 11.1 Architectural Rationale: Why PostgreSQL 16?
In legal, forensic, and safety systems, data integrity, transactional reliability, and deterministic querying are paramount:
- **Relational Integrity**: Enforces foreign key constraints between sessions, entities, chat logs, and legal mappings.
- **ACID Compliance**: Guarantees that multi-table session state transitions complete atomically.
- **Parameterized SQL Security**: Immune to SQL injection vulnerabilities when accessed via SQLAlchemy ORM.
- **Deterministic Search**: Ensures statutory provisions are queried via exact relational indexing rather than probabilistic vector approximations.

## 11.2 Complete 11-Table Schema Architecture
The database schema consists of **7 Core Operational Tables** and **4 Audit/Analytics Tables**:

```
+-------------------------------------------------------------------------------------------------+
|                                 ABHERA DATABASE SCHEMA (11 TABLES)                              |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   +-----------------------+              1:N             +-----------------------+              |
|   |   incident_sessions   |----------------------------->|     chat_messages     |              |
|   |   (UUID PK, timestamps|                              | (message_id, role,    |              |
|   |    status, ip_hash)   |----+                         |  content, created_at) |              |
|   +-----------------------+    |                         +-----------------------+              |
|               |                |                                                                |
|               | 1:N            | 1:N                     +-----------------------+              |
|               v                +------------------------>|   extracted_entities  |              |
|   +-----------------------+    |                         | (entity_id, tag,      |              |
|   | classified_categories |    |                         |  span_text, offsets)  |              |
|   | (category, score,     |    |                         +-----------------------+              |
|   |  threshold, positive) |    |                                                                |
|   +-----------------------+    | 1:1                     +-----------------------+              |
|                                +------------------------>|    collected_slots    |              |
|   +-----------------------+                              | (rel, loc, time, freq,|              |
|   |    legal_provisions   |                              |  platform, evidence)  |              |
|   | (section, act, text,  |                              +-----------------------+              |
|   |  category, penalty)   |                                                                     |
|   +-----------------------+                              +-----------------------+              |
|                                                          |    support_services   |              |
|   +-----------------------+                              | (name, phone, level,  |              |
|   |     audit_logs        |                              |  category, hours)     |              |
|   | (event, metadata, ts) |                              +-----------------------+              |
|   +-----------------------+                                                                     |
|                                                                                                 |
|   [Audit & Analytics]: session_metrics | model_feedback | report_exports | audit_logs           |
+-------------------------------------------------------------------------------------------------+
```

## 11.3 Comprehensive Data Dictionary

### Table 1: `incident_sessions` (Core Session Root)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, `default=uuid4` | Unique session tracking identifier. |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL, default=now()` | Initial session creation timestamp. |
| `updated_at` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL, default=now()` | Last user or model interaction timestamp. |
| `status` | `VARCHAR(32)` | `default='active'` | State: `'active'`, `'completed'`, `'expired'`. |
| `turn_count` | `INTEGER` | `default=0` | Current conversational turn counter. |

### Table 2: `classified_categories` (Model Classification Outputs)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Row identifier. |
| `session_id` | `UUID` | Foreign Key $	o$ `incident_sessions.id` | Associated incident session. |
| `category` | `VARCHAR(64)` | `NOT NULL` | One of 6 harassment categories. |
| `confidence` | `FLOAT` | `NOT NULL` | Sigmoid probability score $[0.0, 1.0]$. |
| `threshold_applied` | `FLOAT` | `NOT NULL` | The cutoff value used for decision. |
| `is_positive` | `BOOLEAN` | `NOT NULL` | Decision indicator (`confidence >= threshold`). |

### Table 3: `extracted_entities` (NER Token Outputs)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Row identifier. |
| `session_id` | `UUID` | Foreign Key $	o$ `incident_sessions.id` | Associated incident session. |
| `entity_type` | `VARCHAR(32)` | `NOT NULL` | Dimension: `PERP`, `REL`, `LOC`, `TIME`, `PLATFORM`. |
| `entity_text` | `TEXT` | `NOT NULL` | Exact extracted character text. |
| `start_offset` | `INTEGER` | `NOT NULL` | Character start position in narrative. |
| `end_offset` | `INTEGER` | `NOT NULL` | Character end position in narrative. |

### Table 4: `collected_slots` (Stateful Question Tracking)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `session_id` | `UUID` | Primary Key, Foreign Key $	o$ `sessions.id` | 1:1 relationship with session. |
| `relationship` | `TEXT` | `NULL` | Populated relationship description. |
| `location` | `TEXT` | `NULL` | Populated location description. |
| `time` | `TEXT` | `NULL` | Populated incident timing. |
| `frequency` | `TEXT` | `NULL` | Populated occurrence frequency. |
| `platform` | `TEXT` | `NULL` | Populated communication platform. |
| `evidence` | `TEXT` | `NULL` | Populated available evidence records. |
| `is_saturated` | `BOOLEAN` | `default=FALSE` | Flag set when required slots are filled. |

### Table 5: `legal_provisions` (Curated Statutory Knowledge)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Provision identifier. |
| `act_name` | `VARCHAR(128)` | `NOT NULL` | Enactment: BNS 2023, IT Act 2000, POSH Act. |
| `section_number` | `VARCHAR(32)` | `NOT NULL` | Exact section number (e.g., "Section 75"). |
| `category` | `VARCHAR(64)` | `NOT NULL` | Primary matching harassment category. |
| `description` | `TEXT` | `NOT NULL` | Summary of prohibited act and offenses. |
| `punishment` | `TEXT` | `NOT NULL` | Prescribed penal sentence, fine, or remedy. |
| `cognizable` | `BOOLEAN` | `NOT NULL` | Cognizability flag under CrPC / BNSS. |

### Table 6: `support_services` (Emergency & Support Infrastructure)
| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Service identifier. |
| `name` | `VARCHAR(128)` | `NOT NULL` | Institution name (e.g., "National Women Helpline"). |
| `phone_number` | `VARCHAR(32)` | `NOT NULL` | Direct telephone dial string (e.g., "181"). |
| `tier_level` | `VARCHAR(32)` | `NOT NULL` | Tier: `'national'`, `'state'`, `'cybercell'`. |
| `category` | `VARCHAR(64)` | `NULL` | Specialized domain or `'general'`. |
| `operational_hours`| `VARCHAR(64)` | `default='24x7'` | Availability window. |

## 11.4 Seed Data Implementation
The database is initialized and populated with verified legal and support data via `backend/app/db/seeds.py`:
- **32 Verified Legal Sections**: Curated across BNS 2023, IT Act, and special women-centric acts.
- **12 Verified Emergency Helplines**: Covering National Emergency (112), Women's Helpline (181), Cyber Crime Portal (1930), and state desks.

## 11.5 Documented Schema Evolution Roadmap
As identified during the architecture audit, the `legal_provisions` table will be upgraded in future migrations with three dedicated columns:
1. `bns_section`: Dedicated string for new Bharatiya Nyaya Sanhita section codes.
2. `punishment_summary`: Concise one-line penalty synopsis for UI card rendering.
3. `bailable`: Explicit boolean flag (`TRUE` = bailable, `FALSE` = non-bailable) to clarify immediate arrest implications.

---


# Chapter 12: Legal Mapping & Anti-Hallucination

## 12.1 The Hallucination Threat in Legal AI
Generative Large Language Models (LLMs) operate via probabilistic next-token prediction. While adept at conversational fluency, LLMs exhibit catastrophic vulnerabilities in legal tech:
- **Fabricated Sections**: Inventing non-existent sections (e.g., *"Under Section 498B of the Indian Penal Code..."* when Section 498B does not exist).
- **Statutory Drift**: Confusing repealed provisions of the Indian Penal Code (IPC 1860) with newly enacted provisions of the Bharatiya Nyaya Sanhita (BNS 2023).
- **Procedural Misdirection**: Advising a survivor that a non-cognizable offense permits immediate police arrest without a magistrate's warrant.

In safety-critical triage, an erroneous legal citation can invalidate an official First Information Report (FIR) or mislead a victim at police desks.

## 12.2 ABHERA's Zero-Hallucination Architecture
ABHERA solves this vulnerability through a **Neuro-Symbolic Architecture**:
1. **Neural Component (BERT)**: Restricted strictly to *classification* (understanding human narrative) and *entity extraction* (pinpointing facts).
2. **Symbolic Component (Relational PostgreSQL Database)**: All legal citations, penal statutes, bailability statuses, and helpline numbers are queried deterministically from verified database records.
3. **Zero Generative Text in Legal Advice**: No generative LLM writes, modifies, or invents legal advice. Every statute presented in the application is an exact verbatim database record curated by legal experts.

```
+-----------------------------------------------------------------------------------+
|                        NEURO-SYMBOLIC SEPARATION OF CONCERNS                      |
|                                                                                   |
|   UNSTRUCTURED NARRATIVE                  STRUCTURED KNOWLEDGE                    |
|   "My husband beat me..."                 PostgreSQL Relational DB                |
|             |                                       |                             |
|             v                                       v                             |
|   +-----------------------+               +-----------------------+               |
|   | BERT Classification   |               | 32 Verified BNS Laws  |               |
|   | & Entity Recognition  |               | & Support Helplines   |               |
|   +-----------+-----------+               +-----------+-----------+               |
|               |                                       |                           |
|               | Positive Categories:                  | Exact SQL Query:          |
|               | [domestic_violence]                   | SELECT * WHERE cat = DV   |
|               +-------------------+-------------------+                           |
|                                   |                                               |
|                                   v                                               |
|               +---------------------------------------+                           |
|               | Deterministic Legal Matcher           |                           |
|               | (Zero LLM Text Generation)            |                           |
|               +---------------------------------------+                           |
|                                   |                                               |
|                                   v                                               |
|               BNS Section 85: Cruelty by Husband                                  |
|               BNS Section 86: Harassment for Dowry                                |
|               PWDVA Section 18: Protection Orders                                 |
+-----------------------------------------------------------------------------------+
```

## 12.3 Legal Knowledge Base: 32 Verified Statutory Provisions
The database curates 32 statutory provisions across Indian criminal and civil jurisprudence:

| Enactment | Section | Key Provision & Legal Scope | Bailability / Cognizability |
| :--- | :--- | :--- | :--- |
| **BNS 2023** | **Section 75** | Sexual Harassment: Unwelcome physical contact, demands for sexual favors, sexually colored remarks. | Cognizable, Bailable |
| **BNS 2023** | **Section 78** | Stalking: Repeatedly following, contacting, or monitoring a woman online despite disinterest. | Cognizable, Bailable (1st offense) |
| **BNS 2023** | **Section 79** | Word, gesture, or act intended to insult the modesty of a woman. | Cognizable, Bailable |
| **BNS 2023** | **Section 85** | Husband or relative of husband subjecting woman to cruelty. | Cognizable, Non-Bailable |
| **BNS 2023** | **Section 86** | Cruelty defined: Harassment with a view to coercing unlawful property/dowry demands. | Cognizable, Non-Bailable |
| **BNS 2023** | **Section 351** | Criminal Intimidation: Threatening injury to person, reputation, or property. | Non-Cognizable, Bailable |
| **IT Act 2000** | **Section 66E** | Violation of privacy: Transmitting/capturing images of private areas without consent. | Cognizable, Bailable |
| **IT Act 2000** | **Section 67** | Publishing or transmitting obscene material in electronic form. | Cognizable, Bailable (1st offense) |
| **IT Act 2000** | **Section 67A** | Publishing or transmitting material containing sexually explicit acts in electronic form. | Cognizable, Non-Bailable |
| **POSH Act 2013** | **Section 3 & 9** | Prevention of sexual harassment at workplace & complaint filing before Internal Committee (IC). | Civil Redressal / Inquiry |
| **PWDVA 2005** | **Section 12 & 18** | Application to Magistrate for Protection Orders, Residence Orders, and Monetary Relief. | Quasi-Criminal / Civil Relief |

## 12.4 Three-Tier Support Service Mapping
Assistance hotlines are organized into three hierarchical emergency tiers:
1. **Tier 1: National Emergency Infrastructure**:
   - `112`: National Emergency Response Support System (Police, Fire, Ambulance).
   - `181`: National Women Helpline (Toll-free 24x7 emergency response, rescue, and counseling).
   - `1090`: Women Powerline (Direct immediate police intervention for stalking and harassment).
2. **Tier 2: Cybercrime Reporting Desks**:
   - `1930`: National Cyber Crime Reporting Portal Helpline.
   - `https://cybercrime.gov.in`: Online portal for filing complaints regarding non-consensual images, cyberstalking, and online fraud.
3. **Tier 3: Specialized Institutional Bodies**:
   - National Commission for Women (NCW) Grievance Cell (`7827170170`).
   - One Stop Crisis Centres (Sakhi Centres) providing integrated medical, legal, and psychological aid.

---


# Chapter 13: Frontend Implementation

## 13.1 Architecture & Component Hierarchy
ABHERA's frontend is developed as a modern, responsive Single-Page Application (SPA) using **React 18**, **TypeScript**, and **Vite**:
- **Design Philosophy**: Trauma-informed UI/UX, prioritizing clarity, privacy, calm color schemes (navy `#0f172a`, blue `#1d4ed8`, soft slate `#f8fafc`), high-contrast typography, and instantaneous feedback.
- **Routing Structure**: Managed via React Router v6 across four dedicated views:
  1. `/` (`HomePage.tsx`): Educational overview, safety assurances, and call-to-action triage launcher.
  2. `/analysis` (`AnalysisPage.tsx`): Core triage workflow integrating narrative input, BERT category rendering, NER tag pills, and the dynamic follow-up chat engine.
  3. `/report` (`ReportPage.tsx`): Interactive preview of the 10-section dossier, statutory legal cards, support hotlines, and instant PDF download.
  4. `/about` (`AboutPage.tsx`): Comprehensive academic documentation, system architecture diagrams, and ethical governance statements.

```
                                  [ App.tsx ]
                                       |
    +------------------+---------------+---------------+------------------+
    |                  |                               |                  |
    v                  v                               v                  v
[ HomePage ]     [ AnalysisPage ]                [ ReportPage ]     [ AboutPage ]
                       |                               |
       +---------------+---------------+       +-------+-------+
       |               |               |       |               |
       v               v               v       v               v
[ IncidentForm ] [ CategoryBadge ] [ ChatInterface ] [ LegalCard ] [ ReportSummary ]
                       |
                       v
                 [ EntityPill ]
```

## 13.2 Core Component Responsibilities

### 1. `IncidentForm.tsx`
- Renders an unconstrained natural language textarea with live character and word counters.
- Enforces client-side validation (minimum 15 characters to prevent empty submissions).
- Displays clear trauma-informed guidance reassuring users about privacy and anonymous processing.

### 2. `CategoryBadge.tsx`
- Renders predicted harassment categories as visually distinctive badges.
- Applies color-coded status styling based on confidence thresholds:
  - Green / Blue (`>= threshold`): Statistically significant positive classification.
  - Muted Gray (`< threshold`): Sub-threshold category (suppressed from primary legal triggers).
- Displays exact percentage confidence scores alongside threshold values for transparency.

### 3. `EntityPill.tsx`
- Displays extracted Named Entities as interactive tag chips categorized by dimension:
  - Blue (`relationship`), Violet (`platform`), Emerald (`location`), Amber (`time_freq`), Rose (`perpetrator`).
- Enables users to visually verify the factual elements extracted from their text.

### 4. `ChatInterface.tsx`
- Implements a conversational dialogue stream for dynamic follow-up questioning.
- Renders assistant inquiries and user responses with animated typing indicators.
- Provides a dedicated **"Skip Question"** button allowing users to bypass uncomfortable queries without stalling the triage pipeline.

### 5. `LegalCard.tsx`
- Renders individual statutory provisions retrieved from PostgreSQL.
- Highlights Section Number (e.g., "Section 75 BNS"), Enactment Name, offense description, and penal consequences.
- Displays prominent visual badges indicating whether the offense is **Cognizable** (arrest without warrant permitted) and **Bailable**.

### 6. `ReportSummary.tsx`
- Provides a full-screen, collapsible overview of the complete incident dossier.
- Integrates the primary **"Download Official PDF Dossier"** trigger calling the client-side `jsPDF` engine.

## 13.3 Client-Side State Management & `sessionStorage`
Safety and privacy requirements strictly prohibit storing incident narratives in persistent browser `localStorage` where shared household computers might expose sensitive disclosures to abusers.
Instead, ABHERA utilizes **React Component State** synchronized with **`sessionStorage`**:
- All session keys (`abhera_session_id`, `abhera_analysis_result`, `abhera_chat_history`) exist only within the active browser tab.
- Closing the browser tab or clicking the "Clear Session / Quick Exit" button instantaneously purges all narrative and entity data from browser memory.

---


# Chapter 14: Report Generation & Evidentiary Dossier

## 14.1 The 10-Section Structured Dossier Architecture
The endpoint `/api/v1/report` compiles a standardized, forensic-quality incident dossier structured into 10 explicit sections:

| Section Index | Dossier Section Name | Evidentiary & Operational Content |
| :--- | :--- | :--- |
| **Section 1** | Incident Header & Metadata | Unique Session UUID, generation timestamp, assessment engine version (`v2.0-bns`). |
| **Section 2** | Primary Incident Narrative | Complete unedited verbatim statement provided by the user. |
| **Section 3** | Classified Harassment Types | Multi-label categories detected, sigmoid confidence scores, applied thresholds, and rationale. |
| **Section 4** | Extracted Factual Entities | Extracted perpetrator identities, relationships, physical locations, dates, and digital platforms. |
| **Section 5** | Follow-Up Transcript | Complete record of dynamic follow-up questions posed and answers provided by the user. |
| **Section 6** | Information Saturation Status | Analysis of populated vs missing evidentiary slots (`relationship`, `location`, `platform`, etc.). |
| **Section 7** | Applicable Legal Framework | Exact statutory citations under Bharatiya Nyaya Sanhita (BNS 2023), IT Act, and special acts. |
| **Section 8** | Emergency Support Services | Actionable contact directory: National 112/181, Cyber Crime Portal 1930, and local desks. |
| **Section 9** | Evidentiary Recommendations | Step-by-step guidance on preserving screenshots, call recordings, CCTV footage, and witness lists. |
| **Section 10** | Legal & Ethical Disclaimer | Clear declaration that the dossier provides automated legal triage, not licensed attorney representation. |

## 14.2 Client-Side Vector PDF Compilation via `jsPDF`
Unlike traditional systems that assemble PDF documents on the backend server (which introduces server compute overhead and transmits private incident text to temporary disk files), ABHERA executes PDF compilation **entirely inside the user's browser**:
- **Library**: `jsPDF 2.5+` paired with `jspdf-autotable 3.8+`.
- **Styling**: Standardized A4 portrait layout with academic margins (15mm), navy primary headers (`#0f172a`), alternating row shading, bordered callouts, and page numbering (`Page X of Y`).
- **Security Guarantee**: Because the PDF is rendered directly from React state into browser memory (`doc.save("ABHERA_Incident_Report.pdf")`), zero intermediate document files are created on any server.

## 14.3 Practical Evidentiary Value for Survivors
The generated PDF dossier serves three practical institutional functions:
1. **First Information Report (FIR) Companion**: Can be handed directly to police station Duty Officers or Women Help Desks, eliminating oral miscommunication and ensuring proper BNS sections are invoked.
2. **Internal Committee (IC) Workplace Complaint**: Provides the structured factual timeline and POSH Act citations required for formal corporate inquiries.
3. **Legal Aid Case Brief**: Enables legal aid advocates and paralegals to rapidly understand the case facts without subjecting the traumatized survivor to repetitive interrogations.

---


# Chapter 15: Complete End-to-End Working Example

## 15.1 Real-World Fictional Scenario
To illustrate the complete operational data flow through all layers of ABHERA, consider the following representative incident narrative:

> *"For the past three weeks, my senior project manager Rohit at our Gurgaon office has been sending me inappropriate personal messages on WhatsApp late at night asking me to visit his apartment. When I politely refused, he threatened to give me a negative performance rating and cancel my upcoming project promotion. Yesterday, he blocked my doorway in the office hallway and touched my shoulder without my consent."*

## 15.2 The 28-Step Step-by-Step System Execution Walkthrough

```
[ PHASE 1: SUBMISSION & PREPROCESSING ]
Step 01: User opens ABHERA web app and pastes narrative into IncidentForm on /analysis.
Step 02: Client validates string length (428 characters, 71 words > 15 character minimum).
Step 03: Frontend dispatches HTTP POST /api/v1/analyze with JSON payload.
Step 04: FastAPI router validates payload against IncidentSubmissionRequest schema.
Step 05: Database opens transaction: creates new row in incident_sessions (UUID: 7a8b...4f1e).

[ PHASE 2: TRANSFORMER CLASSIFICATION ]
Step 06: Narrative text is fed to bert-base-uncased classification tokenizer.
Step 07: Tokenizer outputs token IDs, attention mask, and segment IDs (length: 84 tokens).
Step 08: Model forward pass: 12-layer bidirectional self-attention computes [CLS] embedding.
Step 09: Classification head projects 768-dim pooled output to 6 raw logits:
         [DV: -3.82, SH: +2.95, ST: +1.84, CA: +1.62, WH: +3.10, OV: -1.20].
Step 10: Sigmoid activation computes calibrated probabilities:
         - Domestic Violence    : 0.021  (Threshold 0.73 -> Negative)
         - Sexual Harassment    : 0.950  (Threshold 0.50 -> POSITIVE)
         - Stalking             : 0.863  (Threshold 0.79 -> POSITIVE)
         - Cyber Harassment     : 0.835  (Threshold 0.45 -> POSITIVE)
         - Workplace Harassment : 0.957  (Threshold 0.29 -> POSITIVE)
         - Other Harassment     : 0.231  (Threshold 0.63 -> Negative)
Step 11: Multi-label classification confirms 4 active categories: [SH, ST, CA, WH].
Step 12: Classification results are stored in classified_categories table.

[ PHASE 3: NAMED ENTITY RECOGNITION ]
Step 13: Narrative is passed to fine-tuned BERT-NER token classifier.
Step 14: Sub-word offset mapping aligns token predictions with character boundaries:
         - "Rohit"               -> B-PERP        (Offset 43-48)
         - "senior project manager"-> B/I-PERP_REL  (Offset 20-42)
         - "Gurgaon office"      -> B/I-LOCATION  (Offset 56-70)
         - "past three weeks"    -> B/I-TIME_FREQ (Offset 4-20)
         - "WhatsApp"            -> B-PLATFORM    (Offset 122-130)
Step 15: Extracted entities are stored in extracted_entities table.

[ PHASE 4: DYNAMIC QUESTION ENGINE INQUIRY ]
Step 16: Question Engine initializes collected_slots record.
Step 17: Engine inspects filled slots: [relationship=YES, location=YES, time=YES, platform=YES].
Step 18: Engine detects missing slots: [frequency=PARTIAL, evidence=MISSING].
Step 19: Because primary category is workplace_harassment, evidence is ranked highest priority.
Step 20: Engine generates empathetic follow-up question:
         "Do you currently have any saved records, such as WhatsApp screenshots, emails, or witnesses?"
Step 21: Backend responds with 200 OK containing categories, entities, and question 1.
Step 22: User reviews triage screen and replies in ChatInterface:
         "Yes, I took full screenshots of all WhatsApp chats and have the email where he threatened my rating."
Step 23: Frontend dispatches POST /api/v1/submit-answers.
Step 24: Engine updates collected_slots.evidence = "WhatsApp screenshots and email thread".
Step 25: All required slots are now saturated; engine emits completed: true.

[ PHASE 5: LEGAL MAPPING & REPORT GENERATION ]
Step 26: Legal mapping queries PostgreSQL for provisions matching [SH, ST, CA, WH]:
         - BNS 2023 Section 75 (Sexual Harassment)
         - BNS 2023 Section 78 (Stalking)
         - IT Act 2000 Section 67 (Electronic transmission of obscene messages)
         - POSH Act 2013 Section 3 & 9 (Workplace harassment & Internal Committee complaint)
Step 27: Support mapping retrieves National Helplines (181, 112), Cyber Crime Portal (1930), and Gurugram Women Police Desk.
Step 28: User clicks "Download Official PDF Dossier" on /report; client jsPDF engine compiles and saves the complete 10-section PDF report to the user's downloads folder.
```

---


# Chapter 16: Testing, Quality Assurance & Validation

## 16.1 Testing Methodology & Philosophy
ABHERA enforces an exhaustive automated testing regimen to guarantee that changes to neural models, database schemas, or API schemas never compromise safety or legal accuracy:
- **Zero-Mock Policy for Core Logic**: Model inference, schema validation, slot priority queues, and database mapping functions are tested against real model weights and concrete test databases.
- **Strict Isolation**: Test databases run inside isolated transactions with automatic rollbacks to prevent state leakage.
- **Coverage Standard**: Testing covers nominal user paths, extreme edge cases, malformed payloads, unicode stress tests, and boundary thresholds.

## 16.2 Complete 154-Test Automated Suite
The backend contains **154 automated unit and integration tests** distributed across 11 dedicated test modules:

```
+-----------------------------------------------------------------------------------+
|                        ABHERA AUTOMATED TEST SUITE SUMMARY                        |
|                                                                                   |
|   Module File                  Tests   Focus & Verification Target        Pass %  |
|   --------------------------   -----   ---------------------------------  ------  |
|   test_classification.py        18     BERT multi-label inference & thresholds 100% |
|   test_ner.py                   16     13-tag BIO tags & offset alignment 100%    |
|   test_question_engine.py       18     Slot prioritization & anti-looping 100%    |
|   test_legal_mapping.py         16     32 BNS/IT/POSH laws & accuracy     100%    |
|   test_support_mapping.py       14     3-tier helpline directory lookups  100%    |
|   test_report.py                14     10-section JSON dossier integrity  100%    |
|   test_api.py                   16     End-to-end HTTP routes & statuses  100%    |
|   test_database.py              14     SQLAlchemy ORM & session integrity 100%    |
|   test_session.py               10     Session lifecycle & turn counting  100%    |
|   test_schemas.py               10     Pydantic v2 input/output validation100%    |
|   test_anti_hallucination.py     8     Zero generative text verification  100%    |
|   --------------------------   -----   ---------------------------------  ------  |
|   TOTAL AUTOMATED TESTS        154     COMPLETE CODEBASE VERIFICATION     100%    |
+-----------------------------------------------------------------------------------+
```

## 16.3 Key Test Scenarios & Edge Cases

### 1. Classification Edge Cases (`test_classification.py`)
- **Ultra-Short Inputs**: Verified that 15-character minimum input validation rejects nonsensical single-word strings ("help", "hi").
- **Extreme Multi-Label Overlap**: Verified that complex narratives containing all 6 harassment types correctly trigger all 6 classes simultaneously.
- **Boundary Threshold Accuracy**: Verified that confidence scores exactly equal to the threshold (e.g., $0.7300$ for DV) evaluate as positive, while $0.7299$ evaluates as negative.

### 2. NER Sub-Word Alignment (`test_ner.py`)
- **Punctuation Attachment**: Verified that tokens followed by commas or quotes (e.g., `"Rohit,"`) cleanly isolate `"Rohit"` as the entity span without including punctuation marks.
- **Colloquial Platform Names**: Verified that variations like "Insta", "WApp", "WhatsApp web" properly map to `PLATFORM`.

### 3. Anti-Looping & Termination (`test_question_engine.py`)
- **Forced Turn Cap**: Verified that after 3 turns, the engine emits `completed: true` even if 2 slots remain empty.
- **Repetition Ban**: Verified that answering "husband" in turn 1 makes it mathematically impossible for the engine to query `relationship` in turn 2.

### 4. Anti-Hallucination Guardrails (`test_anti_hallucination.py`)
- **Statutory Verbatim Match**: Verified that every legal section emitted by the service matches character-for-character against the verified database seeds.
- **Zero Third-Party Calls**: Verified that running the entire API test suite dispatches 0 network requests to external LLM providers.

---


# Chapter 17: Complete Development Journey (16 Phases)

The ABHERA system was engineered through a disciplined, phased development lifecycle:

```
+-------------------------------------------------------------------------------------------------+
|                                CHRONOLOGICAL DEVELOPMENT PHASES                                 |
|                                                                                                 |
|   [PHASE 01] Problem Definition & Indian Jurisprudence Analysis (NCRB Data & BNS 2023)         |
|         v                                                                                       |
|   [PHASE 02] System Requirements, 3-Tier Architecture & API Contract Specification              |
|         v                                                                                       |
|   [PHASE 03] Dataset Acquisition, Multi-Label Annotation & Stratified Splitting (1,086 rows)   |
|         v                                                                                       |
|   [PHASE 04] Text Preprocessing & Rigorous Zero-Leakage Audit (Train/Val/Test Isolation)        |
|         v                                                                                       |
|   [PHASE 05] Classical Machine Learning Baselines (TF-IDF + Logistic Regression, SVM, RF)       |
|         v                                                                                       |
|   [PHASE 06] Deep Transformer Exploration: BERT v1 Multi-Label Model (Unweighted BCE)          |
|         v                                                                                       |
|   [PHASE 07] Error Analysis & Innovation: BERT v2 with pos_weight & Threshold Grid Tuning       |
|         v                                                                                       |
|   [PHASE 08] Named Entity Recognition (NER) Development (13 BIO Tags & Sub-Word Alignment)     |
|         v                                                                                       |
|   [PHASE 09] Dynamic Follow-Up Question Engine (State Machine & Priority Queue)                |
|         v                                                                                       |
|   [PHASE 10] PostgreSQL Relational Database Schema & Verified Seed Data Construction            |
|         v                                                                                       |
|   [PHASE 11] FastAPI Backend Implementation, Async Lifespan & Dependency Injection             |
|         v                                                                                       |
|   [PHASE 12] Deterministic Legal Mapping & Anti-Hallucination Knowledge Retrieval              |
|         v                                                                                       |
|   [PHASE 13] React 18 SPA Frontend Development & Trauma-Informed UI/UX Implementation          |
|         v                                                                                       |
|   [PHASE 14] Client-Side Forensic Dossier Generation & jsPDF Report Engine                     |
|         v                                                                                       |
|   [PHASE 15] Comprehensive Automated Testing Suite Construction (154 Pytest Cases)             |
|         v                                                                                       |
|   [PHASE 16] Research Paper Methodology Formulation & Independent Documentation Audit           |
+-------------------------------------------------------------------------------------------------+
```

### Detailed Phase Synopsis

- **Phase 1: Problem Definition & Indian Legal Landscape Analysis**: Conducted empirical analysis of NCRB crime statistics, reporting bottlenecks, and the legal implications of the Bharatiya Nyaya Sanhita (BNS 2023) transition.
- **Phase 2: Architecture & Contract Specification**: Defined the 3-tier decoupled architecture, OpenAPI REST endpoints, and Pydantic v2 data contracts.
- **Phase 3: Dataset Acquisition & Multi-Label Stratification**: Curated 1,086 incident narratives and performed iterative stratified splitting into 760 train, 162 val, and 164 test instances.
- **Phase 4: Preprocessing & Data Leakage Audit**: Constructed preprocessing pipelines and executed automated string-overlap scripts, verifying 0.0% sample leakage.
- **Phase 5: Classical Baselines Benchmark**: Trained TF-IDF baseline models with Logistic Regression, Random Forest, and Linear SVM, setting a baseline Micro-F1 of 88.42% to 89.76%.
- **Phase 6: BERT v1 Exploration**: Fine-tuned standard `bert-base-uncased` with unweighted binary cross-entropy, achieving 94.62% Micro-F1 with 9 false negatives.
- **Phase 7: Innovation: BERT v2 with Weighted BCE & Threshold Optimization**: Implemented positive class weights (`pos_weight`) and per-class threshold tuning, elevating Micro-F1 to **98.09%** and slashing false negatives to 1.
- **Phase 8: Token-Level Named Entity Recognition**: Fine-tuned BERT-NER with sub-word offset alignment across 13 BIO tags, achieving 99.24% Micro-F1.
- **Phase 9: Dynamic Question Engine**: Developed the slot-filling state machine with class-conditioned priority queues and a 3-turn anti-looping cutoff.
- **Phase 10: Relational Database Architecture**: Modeled 11 PostgreSQL tables and populated seed records with 32 verified BNS/IT/POSH laws and 12 emergency services.
- **Phase 11: FastAPI Async Backend**: Implemented async route handlers, startup model weight caching, and CORS middleware.
- **Phase 12: Zero-Hallucination Knowledge Retrieval**: Enforced deterministic parameterized database matching, eliminating generative LLM hallucinations.
- **Phase 13: React 18 Frontend SPA**: Built trauma-informed UI components in React and Tailwind CSS, persisting temporary triage state in `sessionStorage`.
- **Phase 14: Client-Side PDF Compilation**: Integrated `jsPDF` and `jspdf-autotable` to generate official 10-section incident dossiers in browser memory.
- **Phase 15: Automated Quality Assurance**: Authored 154 unit and integration tests across 11 test modules, achieving a 100% pass rate.
- **Phase 16: Research Paper & Audit**: Formulated research methodology for ICMEET 2026 submission and completed an exhaustive multi-stage documentation audit.

---


# Chapter 18: Challenges Faced & Engineering Solutions

During system design and implementation, six formidable engineering challenges arose and were systematically resolved:

```
+-------------------------------------------------------------------------------------------------+
|                                 SIX FORENSIC CASE STUDIES                                       |
+-------------------------------------------------------------------------------------------------+
| 1. Class Imbalance & False Negative Suppression  ==> Weighted BCEWithLogitsLoss (pos_weight)    |
| 2. Sub-Word Tokenization & Span Alignment       ==> Tokenizer Offset Mapping & Masking         |
| 3. Large Language Model Hallucinations          ==> Neuro-Symbolic Relational DB Retrieval      |
| 4. Sub-Optimal Classification Cutoffs           ==> Validation Grid Search Threshold Tuning    |
| 5. Conversational Session State Persistence     ==> PostgreSQL Session Store + sessionStorage   |
| 6. Test Suite Isolation & DB Pollution          ==> Scoped Pytest Transaction Rollback Fixtures |
+-------------------------------------------------------------------------------------------------+
```

### Challenge 1: Severe Class Imbalance & Hazardous False Negatives
- **Problem**: Harassment categories had uneven prevalence (e.g., Other Harassment: 15.8%, Domestic Violence: 24.3%). Standard binary cross-entropy biased predictions toward majority negative decisions, producing 9 dangerous false negatives in BERT v1.
- **Engineering Solution**: Replaced standard `BCELoss` with PyTorch's `BCEWithLogitsLoss` configured with a dynamic positive weight vector $	ext{pos\_weight}_c = (N - P_c) / P_c$, heavily penalizing misses on minority classes and reducing false negatives by 88.9%.

### Challenge 2: WordPiece Sub-Word Tokenization & Entity Span Corruption
- **Problem**: BERT's WordPiece tokenizer fragments words into sub-word tokens (e.g., "WhatsApp" $	o$ `['whats', '##ap', '##p']`). Naive token labeling caused fragmented entity reconstruction and corrupted string spans.
- **Engineering Solution**: Leveraged Hugging Face Tokenizers' `return_offsets_mapping=True`. During training, loss was calculated only on the initial sub-token while masking subsequent tokens with label index `-100`. During inference, character offsets directly sliced the original unedited narrative.

### Challenge 3: Factual Hallucinations in Legal and Safety Advice
- **Problem**: Generative LLMs frequently hallucinate non-existent statutory sections or confuse repealed IPC provisions with new BNS laws, posing catastrophic risks to survivors.
- **Engineering Solution**: Adopted a strict **Neuro-Symbolic Architecture**. Neural networks are restricted exclusively to classification and entity extraction; all statutory sections, bailability tags, and helpline numbers are queried deterministically from verified PostgreSQL records.

### Challenge 4: Sub-Optimal Decision Cutoffs (The 0.50 Uniform Trap)
- **Problem**: A naive 0.50 probability cutoff caused excessive false negatives in subtle workplace harassment cases and false alarms in stalking cases.
- **Engineering Solution**: Implemented systematic grid search optimization over the validation set to calculate class-specific optimal thresholds (DV: 0.73, SH: 0.50, ST: 0.79, CA: 0.45, WH: 0.29, OV: 0.63), boosting Micro-F1 from 94.62% to 98.09%.

### Challenge 5: Session State Drift across Asynchronous Requests
- **Problem**: The dynamic follow-up questioning flow required stateful turn tracking across multiple HTTP requests without requiring user login credentials.
- **Engineering Solution**: Implemented a lightweight, anonymous UUID4 session architecture. The backend persists session turns and slot states in PostgreSQL, while the React frontend maintains session tokens in ephemeral `sessionStorage`.

### Challenge 6: Database State Pollution in Automated Unit Testing
- **Problem**: Running 154 unit tests against a live PostgreSQL database created orphaned rows and caused test interdependence failures.
- **Engineering Solution**: Configured SQLAlchemy transactional test fixtures. Each test executes within an isolated database transaction that automatically issues a `ROLLBACK` upon test completion, guaranteeing 100% test isolation.

---


# Chapter 19: Research Paper and Methodology

## 19.1 Conference Submission Profile
- **Conference**: International Conference on Microelectronics, Electromagnetics and Telecommunication (ICMEET 2026)
- **Paper Title**: *"ABHERA: An Adaptive BERT-Based Neuro-Symbolic Framework for Multi-Label Harassment Classification, Entity Extraction, and Verified Statutory Legal Assistance under Indian Jurisprudence"*
- **Target Track**: Artificial Intelligence, Natural Language Processing, and Computational Social Systems.

## 19.2 Novel Scientific Contributions
1. **First Neuro-Symbolic System for BNS 2023**: ABHERA is the first published legal assistance platform tailored specifically to the Bharatiya Nyaya Sanhita (BNS 2023), mapping unconstrained colloquial incident disclosures to verified statutory sections.
2. **Empirical Proof of pos_weight & Threshold Tuning**: Demonstrates that positive-weighted BCE loss combined with class-specific decision thresholds elevates multi-label harassment detection to 98.09% Micro-F1, outperforming uniform threshold baselines by 3.47%.
3. **Sub-Word Offset BIO Alignment for Indian Vernacular**: Establishes a 99.24% accurate token extraction framework across informal Indian English disclosures.
4. **Deterministic Anti-Hallucination Pipeline**: Formalizes an architectural blueprint for safety-critical AI systems that replaces probabilistic generative LLMs with deterministic relational knowledge retrieval.

## 19.3 Comparative Literature Analysis

| System / Study | Classification Architecture | Multi-Label? | Legal Corpus | Entity Extraction | Anti-Hallucination Guarantee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional Legal Chatbots** | Rule-Based / Dialogflow | No (Single) | Static IPC 1860 | Regex / Keywords | Moderate (Rigid) |
| **Commercial LLMs (GPT-4)** | Generative Transformer | Generative | Mixed / Outdated | Few-Shot Extraction | **None (Hallucination Risk)** |
| **Academic NLP Benchmarks** | RoBERTa / DeBERTa | Yes | US / UK Law | None | None (Research Only) |
| **ABHERA (Proposed)** | Fine-Tuned BERT v2 | **Yes (6 Classes)** | **BNS 2023 + IT Act** | **13-Tag BIO NER** | **100% Deterministic Relational** |

---


# Chapter 20: Future Enhancements & Technical Roadmap

The ABHERA architectural foundation is engineered to accommodate progressive technical upgrades:

```
+-------------------------------------------------------------------------------------------------+
|                                    FUTURE TECHNICAL ROADMAP                                     |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [PHASE 1: LINGUISTIC INCLUSION]                                                               |
|   - Multilingual Transformer Migration (IndicBERT / XLM-RoBERTa)                                |
|   - Support for Hindi, Tamil, Telugu, Bengali, and Code-Mixed "Hinglish"                        |
|                                                                                                 |
|   [PHASE 2: PROGRESSIVE WEB APP & OFFLINE RESILIENCE]                                           |
|   - Service Worker incident caching for low-connectivity rural areas                            |
|   - Client-side quantized ONNX model inference directly in WebAssembly                         |
|                                                                                                 |
|   [PHASE 3: INSTITUTIONAL POLICE & COURT INTEGRATION]                                           |
|   - Automated CCTNS (Crime and Criminal Tracking Network and Systems) XML/JSON export           |
|   - Cryptographic Evidence Hashing (SHA-256) for digital forensic chain-of-custody              |
|                                                                                                 |
|   [PHASE 4: DATABASE SCHEMA EVOLUTION]                                                          |
|   - Migration of 3 planned schema columns: bns_section, punishment_summary, and bailable         |
|   - Direct integration with State Legal Services Authorities (SLSA) pro-bono advocate networks   |
+-------------------------------------------------------------------------------------------------+
```

### 1. Multilingual & Code-Mixed NLP (IndicBERT Migration)
India is a linguistically diverse nation where incident disclosures frequently occur in regional languages or code-mixed vernacular (e.g., Hindi written in Roman script: *"Mera boss mujhe roz threaten karta hai"*). Future releases will incorporate `IndicBERT` and fine-tuned multilingual models to serve non-English speakers across urban and rural demographics.

### 2. Progressive Web App (PWA) & Offline Inference
Victims in rural or unstable network environments require offline accessibility. Compiling the BERT classification model to quantized 8-bit ONNX models running directly within the browser via WebAssembly will enable fully offline triage without active internet connections.

### 3. Cryptographic Evidence Hashing & Forensic Chain of Custody
To ensure incident dossiers withstand scrutiny in criminal court trials, future releases will implement cryptographic SHA-256 evidence hashing. Digital screenshots, call logs, and timestamped narratives will be hashed upon upload, creating an immutable tamper-evident forensic audit trail.

---


# Chapter 21: Technical Interview Preparation (65+ Q&As)

This chapter provides comprehensive, examiner-grade answers across nine specialized technical domains.

---

### Domain 1: Machine Learning & BERT Architecture

#### Q1: What does ABHERA stand for, and what is its primary technical objective?
**Answer**: ABHERA stands for **Adaptive BERT-based Harassment Entity Recognition & Assistance**. Its technical objective is to bridge the gap between unstructured, traumatic natural language harassment disclosures and structured statutory legal relief under Indian law by combining multi-label transformer classification, Named Entity Recognition, an adaptive slot-filling question engine, and a deterministic zero-hallucination legal knowledge base.

#### Q2: Why choose BERT over classical machine learning models like SVM or Random Forest?
**Answer**: Classical models rely on sparse bag-of-words or n-gram TF-IDF representations that ignore token order, syntactic hierarchy, and contextual semantics. In contrast, BERT's multi-head bidirectional self-attention conditions on both left and right context simultaneously. This allows BERT to disambiguate polysemous words (e.g., "blocking a doorway" vs "blocking on Instagram") and understand nuanced disclosures that fail in classical models.

#### Q3: Why is multi-label classification necessary rather than multi-class classification?
**Answer**: Multi-class classification enforces a mutually exclusive single-label constraint via softmax ($\sum p_i = 1$). In real-world harassment, an incident frequently encompasses multiple statutory offenses simultaneously. For example, an ex-spouse sending threatening messages on WhatsApp constitutes Domestic Violence, Stalking, and Cyber Harassment concurrently. Multi-label classification models each category as an independent binary Bernoulli event via independent sigmoids.

#### Q4: What is the underlying architecture of `bert-base-uncased`?
**Answer**: It comprises 12 transformer encoder layers ($L=12$), a hidden embedding size of 768 ($H=768$), and 12 bidirectional self-attention heads ($A=12$), totaling approximately 110 million trainable parameters with a WordPiece vocabulary of 30,522 tokens.

#### Q5: How is the final classification decision produced from BERT?
**Answer**: The raw input tokens pass through the 12 encoder layers. The final hidden state corresponding to the special classification token `[CLS]` ($\mathbf{h}_{	ext{CLS}} \in \mathbb{R}^{768}$) is passed through a Dropout layer ($p=0.2$) into a Linear projection layer ($\mathbf{W} \in \mathbb{R}^{6 	imes 768}, \mathbf{b} \in \mathbb{R}^6$) producing raw logits $\mathbf{z} \in \mathbb{R}^6$. Independent sigmoid functions $\sigma(z_c) = 1 / (1 + e^{-z_c})$ then convert logits into probabilities.

#### Q6: What maximum sequence length was selected, and why?
**Answer**: A maximum sequence length of 256 tokens was chosen. Empirical analysis of the dataset revealed that 99.4% of incident disclosures fell within 256 WordPiece tokens. Capping at 256 avoids the quadratic memory complexity $\mathcal{O}(N^2)$ of 512 tokens while preserving complete incident details.

#### Q7: How does BERT handle out-of-vocabulary (OOV) words and misspellings?
**Answer**: BERT uses the WordPiece tokenization algorithm. Unseen or misspelled words are broken down into known sub-word units prefixed by `##` (e.g., "harasser" $	o$ `['harass', '##er']`), allowing the model to construct semantic representations even for informal vernacular or typos.

#### Q8: Did you train BERT from scratch or fine-tune? Why?
**Answer**: We fine-tuned a pre-trained model. Training a 110M parameter transformer from scratch requires tens of millions of documents and immense compute. Fine-tuning transfers generalized linguistic knowledge from pre-training on BooksCorpus and English Wikipedia, requiring only domain-specific task adaptation on our 1,086 incident corpus.

#### Q9: What is the computational latency of the model during inference?
**Answer**: On modern multi-core x86 CPUs, inference latency averages 450ms to 750ms per narrative. On CUDA GPUs, inference completes in under 45ms.

#### Q10: What are the six harassment categories modeled in ABHERA?
**Answer**: 1. Domestic Violence (`domestic_violence`), 2. Sexual Harassment (`sexual_harassment`), 3. Stalking (`stalking`), 4. Cyber Harassment (`cyber_harassment`), 5. Workplace Harassment (`workplace_harassment`), and 6. Other Harassment / Abuse (`other_harassment`).

---

### Domain 2: Loss Functions, Imbalance & Optimization

#### Q11: What loss function is used in BERT v2, and why?
**Answer**: We use PyTorch's `BCEWithLogitsLoss` with positive class weights (`pos_weight`). It combines a Sigmoid layer and Binary Cross-Entropy loss in a numerically stable single operation using the log-sum-exp trick, while penalizing false negative errors on imbalanced minority classes.

#### Q12: How are the positive class weights (`pos_weight`) calculated mathematically?
**Answer**: For each category $c$, $	ext{pos\_weight}_c = (N - P_c) / P_c$, where $N$ is total training samples (760) and $P_c$ is the positive count for category $c$. For example, for Other Harassment with 121 positives, $	ext{pos\_weight} = (760 - 121) / 121 = 5.281$.

#### Q13: Why did BERT v1 underperform compared to BERT v2?
**Answer**: BERT v1 used unweighted binary cross-entropy and a rigid 0.50 cutoff. Because negative instances outnumber positive instances 4:1 per class, the gradients favored predicting 0, leading to 9 missed violations (false negatives) on subtle harassment descriptions. BERT v2 eliminated 8 of these 9 false negatives.

#### Q14: What optimizer was utilized during training?
**Answer**: AdamW (Adam with Decoupled Weight Decay) with an initial learning rate of $2 	imes 10^{-5}$, $eta_1 = 0.9$, $eta_2 = 0.999$, $\epsilon = 1 	imes 10^{-8}$, and weight decay of $0.01$.

#### Q15: Why is AdamW preferred over standard Adam or SGD?
**Answer**: Standard Adam couples $L_2$ regularization directly with the gradient updates, causing weights with large historical gradients to be regularized less. AdamW decouples weight decay from the gradient-based momentum updates, preventing overfitting in deep transformer layers.

#### Q16: What learning rate schedule was implemented?
**Answer**: A linear warmup schedule for the first 10% of total training steps, followed by linear decay to zero over the remaining 90% of training steps.

#### Q17: Why use per-class threshold optimization instead of a uniform 0.50 cutoff?
**Answer**: Different harassment categories have distinct linguistic manifestations and risk profiles. Workplace harassment often presents subtle, veiled retaliations requiring a sensitive threshold (0.29) to avoid missing severe abuse, whereas stalking requires distinct persistent evidence, performing best at a precise threshold of 0.79.

#### Q18: What is the exact mathematical definition of Micro-F1 and Macro-F1?
**Answer**: 
- **Micro-F1** aggregates all global true positives, false positives, and false negatives across all classes: $	ext{Micro-F1} = 2 \cdot \sum 	ext{TP}_c / (2 \cdot \sum 	ext{TP}_c + \sum 	ext{FP}_c + \sum 	ext{FN}_c)$.
- **Macro-F1** computes the unweighted arithmetic mean of individual class F1-scores: $	ext{Macro-F1} = rac{1}{K} \sum_{c=1}^K 	ext{F1}_c$.

---

### Domain 3: Named Entity Recognition (NER)

#### Q19: What is the BIO tagging scheme used in ABHERA?
**Answer**: BIO stands for Beginning, Inside, and Outside. `B-TAG` indicates the first token of an entity span, `I-TAG` denotes subsequent continuation tokens belonging to that same entity, and `O` denotes background non-entity tokens.

#### Q20: What are the 13 BIO tags implemented in the NER model?
**Answer**: `O`, `B-PERP`, `I-PERP`, `B-VICTIM`, `I-VICTIM`, `B-PERP_REL`, `I-PERP_REL`, `B-LOCATION`, `I-LOCATION`, `B-TIME_FREQ`, `I-TIME_FREQ`, `B-PLATFORM`, and `I-PLATFORM`.

#### Q21: What is the sub-word alignment problem in transformer NER?
**Answer**: BERT's WordPiece tokenizer breaks words into multiple tokens (e.g., `WhatsApp` $	o$ `['whats', '##ap', '##p']`). If only the first token receives the label `B-PLATFORM` and sub-tokens receive `O`, standard string reconstruction outputs broken substrings.

#### Q22: How does ABHERA resolve sub-word token alignment?
**Answer**: We extract character offsets using `return_offsets_mapping=True`. During training, sub-tokens are assigned label `-100` so PyTorch cross-entropy ignores them in gradient calculations. During inference, entity spans are reconstructed directly by slicing character offsets from the original raw input string.

#### Q23: What overall Micro-F1 did the BERT-NER model achieve?
**Answer**: An overall Micro-F1 of **99.24%** with 98.92% precision and 99.56% recall.

#### Q24: Which entity categories achieved 100% recall?
**Answer**: Relationship (`PERP_REL`), Time/Frequency (`TIME_FREQ`), and Digital Platform (`PLATFORM`) all achieved 100.00% recall on the test set.

#### Q25: Why did Location achieve a slightly lower recall of 92.31%?
**Answer**: Location descriptions exhibit the highest vocabulary variance in natural language (e.g., "stairwell near 3rd floor cafeteria"), occasionally overlapping with common background nouns.

#### Q26: How are extracted entities passed to downstream components?
**Answer**: Extracted entity spans are normalized, deduplicated, and mapped into a structured Python dictionary that directly populates the conversational slot tracker and database tables.

---

### Domain 4: Dynamic Question Engine

#### Q27: What is the purpose of the Dynamic Question Engine?
**Answer**: Survivors often omit vital evidentiary facts during initial emotional disclosures. The Question Engine identifies missing legal elements and asks empathetic follow-up questions to prepare a complete incident report without overwhelming the user.

#### Q28: What six information slots does the engine track?
**Answer**: `relationship`, `location`, `time`, `frequency`, `platform`, and `evidence`.

#### Q29: How does the priority queue select the next question?
**Answer**: The priority queue is conditioned on the primary predicted harassment category. For instance, if `workplace_harassment` is detected, `relationship` and `evidence` are ranked at the top of the inquiry queue.

#### Q30: How does the engine prevent infinite loops or repetitive questioning?
**Answer**: Through three deterministic safeguards: a hard turn cap of 3 rounds, a strict non-repetition filter that permanently blocks already filled or extracted slots, and an automated skip-handler.

#### Q31: What happens if a user clicks "Skip Question"?
**Answer**: The slot is marked as `SKIPPED`, the turn counter increments, and the engine evaluates the next highest-priority unfilled slot without repeating the skipped query.

#### Q32: When does the questioning sequence terminate?
**Answer**: Termination occurs when either: 1) The conversational turn count reaches 3, or 2) All required slots are saturated with validated information.

#### Q33: How does the engine generate questions without an LLM?
**Answer**: Questions are generated using curated, trauma-informed question templates indexed by slot and conditioned on the active harassment class.

---

### Domain 5: Legal Mapping & Anti-Hallucination

#### Q34: What is the fundamental risk of using generative LLMs for legal advice?
**Answer**: LLMs suffer from probabilistic hallucinations—inventing non-existent legal sections, citing repealed laws (e.g., IPC 1860 instead of BNS 2023), or misrepresenting bailability status, which can mislead a survivor or cause a police complaint to be rejected.

#### Q35: How does ABHERA guarantee zero hallucinations in legal mapping?
**Answer**: Through a strict Neuro-Symbolic architecture. Neural networks are restricted to classification and entity extraction. All legal citations, statutes, and penalties are queried deterministically from a verified PostgreSQL relational database using parameterized SQLAlchemy ORM queries.

#### Q36: How many legal provisions are curated in the ABHERA database?
**Answer**: 32 verified statutory provisions across BNS 2023, IT Act 2000, POSH Act 2013, and PWDVA 2005.

#### Q37: What is BNS Section 75?
**Answer**: Section 75 of the Bharatiya Nyaya Sanhita (2023) penalizes Sexual Harassment, covering unwelcome physical contact, demands for sexual favors, and sexually colored remarks (equivalent to former IPC Section 354A).

#### Q38: What is BNS Section 78?
**Answer**: Section 78 of the Bharatiya Nyaya Sanhita (2023) penalizes Stalking, covering repeated physical following, monitoring, or contacting a woman online despite clear disinterest.

#### Q39: What is BNS Section 85 & 86?
**Answer**: Sections 85 and 86 of the Bharatiya Nyaya Sanhita (2023) penalize Cruelty by a Husband or his relatives, including harassment for unlawful dowry demands (equivalent to former IPC Section 498A).

#### Q40: What are the three tiers of emergency support services?
**Answer**: 1. Tier 1: National Emergency (112, 181, 1090), 2. Tier 2: Cyber Crime Portal (1930), 3. Tier 3: Specialized institutional bodies (NCW, Sakhi One Stop Centres).

---

### Domain 6: System Architecture & Backend (FastAPI)

#### Q41: Why use FastAPI instead of Flask or Django?
**Answer**: FastAPI natively supports asynchronous I/O via `async`/`await`, handles high concurrent throughput, automatically generates OpenAPI 3.0 documentation, and utilizes Pydantic v2 for lightning-fast C-compiled data validation.

#### Q42: How does the backend prevent reloading heavy BERT models on every request?
**Answer**: By utilizing FastAPI's `lifespan` context manager. Models are instantiated once during application startup, cached in global application memory, and shared across all request worker threads.

#### Q43: How is CORS configured on the backend?
**Answer**: Via FastAPI's `CORSMiddleware`, configured to allow explicit frontend origins (e.g., `http://localhost:5173`) while permitting standard HTTP verbs (`GET`, `POST`, `OPTIONS`).

#### Q44: What is the role of Pydantic v2 schemas in ABHERA?
**Answer**: Pydantic schemas enforce strict runtime data contracts on all incoming JSON payloads, validating types, field boundaries, and constraints before data reaches the service logic.

#### Q45: How does the backend handle unexpected model or database exceptions?
**Answer**: Global exception handlers intercept unhandled errors, log forensic stack traces internally, and return structured RFC 7807 JSON error responses with sanitized user-facing messages.

#### Q46: How are database transactions handled in the backend?
**Answer**: Transactions are managed via scoped SQLAlchemy sessions injected using FastAPI's dependency injection (`Depends(get_db)`), ensuring connections automatically close after each request.

#### Q47: What is the function of the `/health` endpoint?
**Answer**: It acts as a Kubernetes / Docker liveness and readiness probe, verifying that the database connection pool is active and both transformer models are loaded in memory.

#### Q48: How are sessions identified and tracked across requests?
**Answer**: Each session is assigned an immutable UUID4 token generated by the server upon initial narrative submission and returned to the client.

---

### Domain 7: Database & Data Modeling (PostgreSQL)

#### Q49: Why choose PostgreSQL over a NoSQL database like MongoDB?
**Answer**: Legal and forensic triage demands ACID compliance, strict relational foreign key constraints, deterministic querying, and zero schema drift across multi-table transactional workflows.

#### Q50: How many tables exist in the ABHERA PostgreSQL schema?
**Answer**: 11 tables: 7 core operational tables and 4 audit/analytics tables.

#### Q51: What are the 7 core operational tables?
**Answer**: `incident_sessions`, `chat_messages`, `classified_categories`, `extracted_entities`, `collected_slots`, `legal_provisions`, and `support_services`.

#### Q52: What are the 4 audit and analytics tables?
**Answer**: `session_metrics`, `model_feedback`, `report_exports`, and `audit_logs`.

#### Q53: What is the relationship between `incident_sessions` and `collected_slots`?
**Answer**: A 1:1 relationship where `collected_slots.session_id` acts as both Primary Key and Foreign Key referencing `incident_sessions.id`.

#### Q54: What is the relationship between `incident_sessions` and `classified_categories`?
**Answer**: A 1:N relationship allowing multiple harassment categories (up to 6) to be associated with a single incident session.

#### Q55: What three columns are planned for the upcoming database migration?
**Answer**: `bns_section` (dedicated string), `punishment_summary` (short synopsis), and `bailable` (boolean flag).

---

### Domain 8: Frontend & User Experience

#### Q56: Why choose Vite over Create React App (CRA)?
**Answer**: Vite uses native ES modules during development, providing instant server start and lightning-fast Hot Module Replacement (HMR) without costly whole-bundle recompilation.

#### Q57: How is application state maintained across user navigation?
**Answer**: Using React state synchronized with browser `sessionStorage`. This maintains data across page refreshes while ensuring all records are wiped when the browser tab is closed.

#### Q58: Why avoid browser `localStorage` in ABHERA?
**Answer**: In domestic or workplace abuse scenarios, abusers may inspect the survivor's shared computer. `localStorage` persists indefinitely, creating severe safety risks. `sessionStorage` purges automatically upon tab closure.

#### Q59: How is the PDF dossier generated on the client side?
**Answer**: Using `jsPDF` and `jspdf-autotable`. The report is compiled directly in browser memory and triggered as a client-side download without transmitting private incident details to server disk storage.

#### Q60: What design choices support trauma-informed UX?
**Answer**: A calm navy and soft blue color palette, prominent reassurance banners regarding privacy, a "Skip Question" button, and clear next-step legal cards.

---

### Domain 9: System Security, Ethics & Privacy

#### Q61: Does ABHERA transmit user text to any external APIs?
**Answer**: No. All classification, entity recognition, and legal mapping operations run strictly on locally hosted models and internal PostgreSQL databases. Zero data is transmitted to third-party APIs.

#### Q62: Does ABHERA replace licensed legal advocates?
**Answer**: No. ABHERA is an automated legal triage and evidentiary assistance tool designed to inform survivors and aid in drafting complaints. Every screen and generated report carries an explicit statutory disclaimer.

#### Q63: How are SQL injection attacks prevented?
**Answer**: All database interactions use SQLAlchemy 2.0 ORM with parameterized query bindings, ensuring user text is never directly interpolated into raw SQL strings.

#### Q64: How is data leakage audited between training and evaluation splits?
**Answer**: Using automated string hash matching and Levenshtein similarity audits, verifying 0.0% overlap between train, validation, and test datasets.

#### Q65: How many automated unit tests protect the codebase against regression?
**Answer**: 154 automated unit and integration tests across 11 modules, with a 100% pass rate.

---


# Chapter 22: Project Presentation and Viva Defense

## 22.1 Presentation Pitches

### 1-Minute Elevator Pitch
> *"Good morning, respected examiners. Across India, millions of women experience harassment every day—ranging from domestic violence to workplace abuse and cyberstalking. Yet, less than 20% ever report these crimes due to trauma, social stigma, and confusion surrounding Indian penal laws—especially with the recent transition to the Bharatiya Nyaya Sanhita (BNS 2023). While commercial generative AI can answer questions, it frequently hallucinates non-existent legal sections, risking survivors' legal standing. To solve this, we engineered **ABHERA**—an Adaptive BERT-based Harassment Entity Recognition and Assistance system. ABHERA accepts free-form incident descriptions, accurately detects overlapping harassment categories using fine-tuned BERT v2 with an industry-leading **98.09% Micro-F1**, extracts critical evidentiary entities using BERT-NER at **99.24% F1**, asks dynamic follow-up questions to fill missing facts, and maps the incident deterministically to 32 verified BNS statutes and support helplines without a single AI hallucination. It produces an official evidentiary PDF dossier directly in the browser. In short, ABHERA transforms unstructured trauma into structured, actionable legal justice."*

### 3-Minute Comprehensive Project Pitch
> *"Respected panel of examiners, our project is ABHERA: Adaptive BERT-based Harassment Entity Recognition and Assistance. Our research addresses three acute societal and technical challenges: First, the acute reporting bottleneck where victims struggle to articulate incidents into legal terms; second, the legal ambiguity created by the July 2024 implementation of the Bharatiya Nyaya Sanhita (BNS 2023); and third, the critical hazard of Large Language Model hallucinations in safety-critical domains.*
>
> *To solve this, ABHERA implements an end-to-end neuro-symbolic framework. When a survivor provides an unstructured incident narrative, our application layer routes the text to two specialized transformer models fine-tuned on 1,086 carefully curated Indian harassment cases. The first model is our champion BERT v2 multi-label classifier. By implementing positive-class weighted Binary Cross-Entropy loss to combat severe class imbalance, combined with grid-search threshold optimization across all six harassment classes, our model attains a Micro-F1 of 98.09% on held-out test data, outperforming classical TF-IDF baselines by nearly 10% and reducing false negatives from nine to just one. Simultaneously, our BERT-NER model extracts factual anchors—perpetrators, relationships, platforms, and locations—using a 13-tag BIO schema with sub-word offset alignment, achieving 99.24% Micro-F1.*
>
> *Because initial disclosures often omit vital legal elements, our Dynamic Question Engine analyzes populated versus missing informational slots. Conditioned on the primary predicted harassment category, it dynamically ranks missing slots and poses empathetic follow-up questions, strictly enforcing anti-looping rules with a 3-turn cutoff. Once saturated, the system performs deterministic, zero-hallucination querying against our PostgreSQL repository of 32 verified statutory provisions across BNS 2023, the IT Act, and the POSH Act, alongside a 3-tier emergency support directory. Finally, the client-side React frontend compiles a standardized 10-section evidentiary PDF dossier using jsPDF—completely within browser memory to protect user privacy. Validated by 154 automated unit tests, ABHERA proves that combining deep transformer models with deterministic relational knowledge provides the safe, reliable, and empathetic future for legal-tech."*

### 5-Minute Technical Defense
*(Use this expanded format when delivering a comprehensive project presentation, walking through Architecture, Mathematics, Baselines, and Live Demonstration).*

---

## 22.2 Viva Defense: Tough Examiner Questions & Model Answers

#### Examiner: "Why did you build your own BERT model instead of just calling the OpenAI GPT-4 API with a prompt?"
**Model Defense**:
> *"Respected examiner, we deliberately avoided commercial LLM APIs for three critical reasons: First, **Hallucination Risk**: GPT-4 is a probabilistic autoregressive model. In legal and safety contexts, hallucinating a non-existent penal section (e.g., citing Section 498B or confusing repealed IPC with BNS) invalidates police complaints and misleads victims. Our neuro-symbolic architecture guarantees 100% deterministic accuracy by retrieving laws strictly from a verified database. Second, **Privacy and Data Sovereignty**: Victims disclose deeply personal, traumatic incidents. Transmitting unencrypted domestic or workplace abuse narratives to third-party US cloud servers violates basic privacy ethics. ABHERA runs self-contained on local infrastructure with zero external telemetry. Third, **Cost and Latency**: LLM API calls cost tokens and incur multi-second latencies, whereas our fine-tuned BERT models execute on CPU in under 700 milliseconds with zero ongoing inference cost."*

#### Examiner: "Your test set has only 164 samples. Isn't that too small to claim 98.09% F1?"
**Model Defense**:
> *"While 164 samples is the number of incident narratives, multi-label classification evaluates six independent binary decisions per narrative. Therefore, our test set encompasses **984 distinct binary classification decisions**. Furthermore, our dataset was split using rigorous multi-label iterative stratification, and our automated data leakage audit proved that there is 0.0% overlap between train and test splits. Across those 984 decisions, the model correctly identified 180 true positives and 797 true negatives, committing only 6 false positives and a single false negative."*

#### Examiner: "Can you explain that single false negative (`TEST_0050`)?"
**Model Defense**:
> *"Certainly. Sample `TEST_0050` described an incident during an annual performance review where a team lead implied promotions require 'flexibility outside office hours' and subsequently reassigned client accounts. The ground truth was labeled both Workplace Harassment and Sexual Harassment. Our model correctly predicted Sexual Harassment with 78% confidence (well above the 0.50 threshold), but Workplace Harassment scored 0.24, just below the 0.29 threshold, due to formal corporate vocabulary. Crucially, because Sexual Harassment was correctly triggered, the downstream legal mapping still successfully retrieved BNS Section 75 and POSH Act workplace provisions, ensuring the survivor was not deprived of legal remedy."*

#### Examiner: "Why not use a standard 0.50 threshold across all categories?"
**Model Defense**:
> *"Using a uniform 0.50 threshold assumes that the confidence distribution and risk penalty across all classes are identical. In harassment triage, missing a subtle workplace retaliation carries a high real-world penalty, requiring a sensitive 0.29 cutoff. Conversely, stalking carries distinct criteria of persistent pursuit, where a 0.79 cutoff prevents false accusations. Our grid search optimization on the validation set proved that class-specific thresholds increased Micro-F1 by 3.47% and reduced false negatives from nine to one."*

---


# Chapter 23: Quick Revision Cheatsheet & Summary

## 23.1 Master Numbers Cheatsheet

| Parameter | Exact Value | Verification Source |
| :--- | :--- | :--- |
| **Total Incident Dataset Samples** | **1,086** | `ml/datasets/raw/harassment_dataset.csv` |
| **Training Split Samples** | **760 (70.0%)** | `ml/datasets/processed/train.json` |
| **Validation Split Samples** | **162 (14.9%)** | `ml/datasets/processed/val.json` |
| **Test Split Samples** | **164 (15.1%)** | `ml/datasets/processed/test.json` |
| **Total Test Binary Decisions** | **984** ($164 	imes 6$) | Test set evaluation matrix |
| **Test True Positives (TP)** | **180** | Test set evaluation matrix |
| **Test False Positives (FP)** | **6** | Test set evaluation matrix |
| **Test False Negatives (FN)** | **1 (TEST_0050)** | Test set evaluation matrix |
| **Test True Negatives (TN)** | **797** | Test set evaluation matrix |
| **BERT v2 Micro-Precision** | **96.77%** | `180 / 186` |
| **BERT v2 Micro-Recall** | **99.45%** | `180 / 181` |
| **BERT v2 Micro-F1 Score** | **98.09%** | Champion model evaluation |
| **BERT v2 Macro-F1 Score** | **97.45%** | Arithmetic mean across 6 classes |
| **BERT-NER Micro-F1 Score** | **99.24%** | Token classification test report |
| **Automated Backend Tests** | **154** (100% Pass) | `backend/tests/` (11 modules) |
| **Curated Statutory Provisions** | **32** verified sections | `legal_provisions` database table |
| **Curated Emergency Services** | **12** verified hotlines | `support_services` database table |
| **Maximum Conversational Turns** | **3** | `question_engine.py` turn cap |
| **Information Tracking Slots** | **6** | `collected_slots` database table |
| **Relational Database Tables** | **11** (7 core + 4 audit) | PostgreSQL 16 schema |

## 23.2 Per-Class Optimal Thresholds

```
+-----------------------------------------------------------------------------+
| Category Code          Category Name            Threshold    Sensitivity     |
| -------------------    ---------------------    ---------    -----------     |
| domestic_violence      Domestic Violence          0.73       Specific        |
| sexual_harassment      Sexual Harassment          0.50       Balanced        |
| stalking               Stalking                   0.79       High Precision  |
| cyber_harassment       Cyber Harassment           0.45       Sensitive       |
| workplace_harassment   Workplace Harassment       0.29       Highly Sensitive|
| other_harassment       Other Abuse                0.63       Conservative    |
+-----------------------------------------------------------------------------+
```

## 23.3 Baseline Comparison Snapshot

| Model Architecture | Features | Micro-F1 | Macro-F1 | Key Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF (3K) | 88.42% | 87.12% | Ignores word order & syntactic context |
| **Random Forest** | TF-IDF (3K) | 84.18% | 82.90% | Poor recall on complex minority classes |
| **Linear SVM** | TF-IDF (3K) | 89.76% | 88.64% | Sparse representation misses slang/typos |
| **BERT v1 (Baseline)** | [CLS] dense | 94.62% | 93.85% | 9 false negatives due to class imbalance |
| **BERT v2 (ABHERA)** | [CLS] + pos_weight | **98.09%** | **97.45%** | SOTA champion model; 1 false negative |

---
