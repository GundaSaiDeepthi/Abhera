# ABHERA — MASTER PROJECT KNOWLEDGE, TRAINING & VIVA GUIDE

**Project:** ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance  
**Location:** `D:\Abhera(Mini)`  
**Scope:** Complete Master Documentation for Learning, Paper Writing, Viva Preparation, Technical Defense, and Architectural Understanding  
**Source-of-Truth Policy:** 100% Based on Actual Codebase Inspection of `D:\Abhera(Mini)`. No source code, models, or datasets modified.

---

## TABLE OF CONTENTS

1. [PART 1 — PROJECT OVERVIEW](#part-1--project-overview)
2. [PART 2 — COMPLETE SYSTEM ARCHITECTURE](#part-2--complete-system-architecture)
3. [PART 3 — COMPLETE END-TO-END TABLE](#part-3--complete-end-to-end-table)
4. [PART 4 — DATASET](#part-4--dataset)
5. [PART 5 — DATA VALIDATION](#part-5--data-validation)
6. [PART 6 — TEXT PREPROCESSING](#part-6--text-preprocessing)
7. [PART 7 — FEATURE EXTRACTION](#part-7--feature-extraction)
8. [PART 8 — FEATURE SELECTION](#part-8--feature-selection)
9. [PART 9 — CLASSICAL BASELINES](#part-9--classical-baselines)
10. [PART 10 — BERT CONCEPT](#part-10--bert-concept)
11. [PART 11 — BERT INPUT PROCESSING](#part-11--bert-input-processing)
12. [PART 12 — BERT ARCHITECTURE](#part-12--bert-architecture)
13. [PART 13 — MULTI-LABEL CLASSIFICATION](#part-13--multi-label-classification)
14. [PART 14 — LOSS FUNCTION](#part-14--loss-function)
15. [PART 15 — TRAINING](#part-15--training)
16. [PART 16 — BERT V1](#part-16--bert-v1)
17. [PART 17 — BERT V1 ERROR ANALYSIS](#part-17--bert-v1-error-analysis)
18. [PART 18 — BERT V2](#part-18--bert-v2)
19. [PART 19 — THRESHOLD TUNING](#part-19--threshold-tuning)
20. [PART 20 — FINAL BERT RESULTS](#part-20--final-bert-results)
21. [PART 21 — BERT V1 VS V2](#part-21--bert-v1-vs-v2)
22. [PART 22 — BASELINE VS BERT](#part-22--baseline-vs-bert)
23. [PART 23 — NER](#part-23--ner)
24. [PART 24 — NER PERFORMANCE](#part-24--ner-performance)
25. [PART 25 — QUESTION ENGINE](#part-25--question-engine)
26. [PART 26 — SESSION MANAGEMENT](#part-26--session-management)
27. [PART 27 — DATABASE](#part-27--database)
28. [PART 28 — LEGAL MAPPING](#part-28--legal-mapping)
29. [PART 29 — SUPPORT MAPPING](#part-29--support-mapping)
30. [PART 30 — ANTI-HALLUCINATION](#part-30--anti-hallucination)
31. [PART 31 — REPORT GENERATION](#part-31--report-generation)
32. [PART 32 — REPORT API](#part-32--report-api)
33. [PART 33 — FRONTEND](#part-33--frontend)
34. [PART 34 — PDF GENERATION](#part-34--pdf-generation)
35. [PART 35 — TESTING](#part-35--testing)
36. [PART 36 — LEAKAGE AUDIT](#part-36--leakage-audit)
37. [PART 37 — END-TO-END VALIDATION](#part-37--end-to-end-validation)
38. [PART 38 — FINAL MODEL PROMOTION](#part-38--final-model-promotion)
39. [PART 39 — FORMULAS](#part-39--formulas)
40. [PART 40 — ALGORITHMS / PSEUDOCODE](#part-40--algorithms--pseudocode)
41. [PART 41 — COMPLETE WORKFLOW EXAMPLE](#part-41--complete-workflow-example)
42. [PART 42 — ERROR HANDLING](#part-42--error-handling)
43. [PART 43 — SECURITY / PRIVACY](#part-43--security--privacy)
44. [PART 44 — LIMITATIONS](#part-44--limitations)
45. [PART 45 — FUTURE ENHANCEMENTS](#part-45--future-enhancements)
46. [PART 46 — PAPER-READY METHODOLOGY](#part-46--paper-ready-methodology)
47. [PART 47 — PAPER-READY RESULTS](#part-47--paper-ready-results)
48. [PART 48 — VIVA PREPARATION](#part-48--viva-preparation)
49. [PART 49 — "WHY DID WE USE THIS?" TABLE](#part-49--why-did-we-use-this-table)
50. [PART 50 — "WHAT IF THEY ASK..." VIVA SECTION](#part-50--what-if-they-ask-viva-section)
51. [PART 51 — COMMANDS](#part-51--commands)
52. [PART 52 — PROJECT FILE MAP](#part-52--project-file-map)
53. [PART 53 — SOURCE TRACEABILITY](#part-53--source-traceability)
54. [PART 54 — OBSERVED DISCREPANCIES](#part-54--observed-discrepancies)
55. [PART 55 — FINAL CHEAT SHEET](#part-55--final-cheat-sheet)
56. [PART 56 — FINAL "LEARN THIS IN ORDER"](#part-56--final-learn-this-in-order)

---

# PART 1 — PROJECT OVERVIEW

### Project Title
**ABHERA — Adaptive BERT-based Harassment Entity Recognition & Assistance**

### Meaning of ABHERA
"Abhera" (derived from Sanskrit for "fearless") stands for **A**daptive **B**ERT-based **H**arassment **E**ntity **R**ecognition & **A**ssistance. It signifies empowering survivors of gender-based harassment to report incidents fearlessly by providing immediate, accurate legal guidance and verified support helplines without fear of exposure, privacy violation, or legal hallucinations.

### Problem Statement
Survivors of gender-based harassment and violence face severe challenges in identifying their legal rights, navigating complex legal terminology (IPC/BNS statutory provisions), locating immediate local emergency support helplines, and articulating unstructured trauma narratives into legally relevant incident reports. Existing commercial conversational AI platforms frequently generate fabricated ("hallucinated") legal advice or fail to capture compound incidents (e.g. domestic violence occurring simultaneously with online blackmail).

### Motivation
To create an end-to-end, privacy-preserving, transparent artificial intelligence framework that combines fine-tuned deep learning narrative understanding with deterministic database-grounded legal retrieval to deliver zero-hallucination survivor assistance.

### Objective
1. Multi-label classification of complex survivor incident narratives into 6 statutory categories (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
2. Named Entity Recognition (NER) to extract key evidentiary details (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`).
3. Dynamic questionnaire dialogue to fill missing evidentiary gaps without repetitive or off-topic questions.
4. Deterministic database grounding against verified PostgreSQL legal (`laws`) and support helpline (`support_services`) tables.
5. Automated generation of executive incident summary reports with client-side PDF export.

### Target Users
Survivors of harassment, legal aid volunteers, non-governmental support organizations (NGOs), legal counselors, and institutional harassment reporting committees (e.g. POSH IC committees).

### Main Use Case
A survivor enters an initial narrative in plain English (e.g., *"My husband hit me at home and threatens to post my private photos online"*). ABHERA analyzes the text, identifies compound categories (`DV` + `CA`), extracts entities (`PERP_REL`: husband, `LOCATION`: home), asks context-aware follow-up questions to assess safety and platform details, maps exact applicable statutory sections (e.g. IPC Section 498A / BNS Section 85, IT Act Section 66E) and local emergency helplines from PostgreSQL, and generates a structured downloadable PDF report.

### System Non-Claims (What the System Does NOT Claim To Do)
- ABHERA does **NOT** provide binding legal advice or replace formal legal counsel.
- ABHERA does **NOT** generate legal text dynamically using LLMs (preventing legal hallucinations).
- ABHERA does **NOT** auto-submit formal police complaints or FIRs to law enforcement agencies directly.

### Key Features
- Multi-label BERT classifier with label-specific decision threshold tuning.
- Fine-tuned BERT-NER token classifier with character offset span alignment.
- Rule-based dynamic question engine with priority-driven requirement resolution.
- Anti-hallucination gating layer verifying database provenance before inclusion.
- Structured PDF generator using master design tokens and client-side rendering.

### Major Technologies
- **Frontend:** React 18, Vite, Lucide Icons, jsPDF.
- **Backend:** FastAPI, Pydantic, SQLAlchemy, PostgreSQL.
- **Machine Learning / NLP:** PyTorch, HuggingFace Transformers (`bert-base-uncased`), Scikit-learn, NumPy.

### Explanations of ABHERA

#### A. 30-Second Explanation
> "ABHERA is an AI-powered conversational system for survivors of harassment. It takes unstructured incident descriptions, uses fine-tuned BERT models to classify the types of harassment and extract key evidentiary details, asks necessary follow-up questions, and retrieves verified legal rights and support helplines from a database to produce a structured incident summary report."

#### B. 1-Minute Explanation
> "When survivors experience harassment, understanding their legal rights and finding help is overwhelming. ABHERA solves this by combining deep learning with strict database verification. A survivor types what happened. ABHERA uses a fine-tuned BERT multi-label classifier to detect single or compound incident categories like Domestic Violence or Cyber Abuse. Concurrently, a BERT-NER model extracts entities like perpetrator relationship and location. If details are missing, a dynamic question engine asks context-aware follow-ups. Finally, an anti-hallucination layer looks up verified IPC/BNS statutory sections and helplines in PostgreSQL and compiles a complete downloadable PDF report."

#### C. 3-Minute Explanation
> "ABHERA addresses critical gaps in survivor support technology: legal uncertainty, emotional trauma articulation, and AI legal hallucinations. 
> 
> Architecturally, it consists of a React frontend, a FastAPI microservice backend, PostgreSQL storage, and two fine-tuned BERT transformer models. 
> 
> First, raw narratives are ingested by a BERT multi-label classifier using positive-class weighted binary cross-entropy loss (`BCEWithLogitsLoss`) and six validation-tuned decision thresholds. This achieves a 98.09% Micro-F1 score and 95.73% Exact Match accuracy.
> 
> Second, a token-level BERT-NER model extracts evidentiary entity spans (like perpetrator relationship, location, frequency, and communication platform) with a 99.24% entity-level Micro-F1.
> 
> Third, a dynamic question engine evaluates required versus known information dimensions for active categories, dynamically prioritizing questions to fill missing gaps.
> 
> Fourth, the Anti-Hallucination Service maps predicted categories to verified statutory laws and support helplines stored in PostgreSQL. Crucially, the AI models only predict categories and extract entities—all legal text and contact numbers come verbatim from verified database tables.
> 
> Finally, all 10 conceptual sections are assembled into a structured JSON report and rendered in the frontend with client-side PDF export capabilities."

#### D. Technical Explanation
> "ABHERA implements a hybrid neuro-symbolic architecture for multi-label harassment classification, entity extraction, and legal knowledge retrieval. 
> 
> The classification pipeline fine-tunes `bert-base-uncased` (110M parameters) over 6 incident categories ($\mathbf{y} \in \{0, 1\}^6$) using PyTorch. To handle multi-label dependencies and severe class imbalance, the loss function is configured as `BCEWithLogitsLoss` with class positive-weighting $w_k = (N - N_k)/N_k$. Post-training, optimal decision thresholds $\tau_k \in [0.10, 0.90]$ are fitted on validation probabilities to optimize per-label F1 scores before evaluation on a held-out test set ($N=164$). 
> 
> The entity extraction pipeline uses `AutoModelForTokenClassification` over 13 BIO token tags, utilizing tokenizer offset mappings to align subword predictions back to original character spans. 
> 
> The conversational state machine tracks sessions across four lifecycle phases: `ACTIVE`, `QUESTIONING`, `PROCESSING`, and `COMPLETED`. Known entities are extracted via NER and regex heuristics, and missing dimensions are resolved via a priority-scored question selector. 
> 
> The legal mapping module executes parameterized ORM queries against PostgreSQL `laws` and `support_services` tables. An anti-hallucination validator verifies exact string match against database rows before serializing the final 10-section report schema to the `reports` table."

---

# PART 2 — COMPLETE SYSTEM ARCHITECTURE

```
User
  │
  ▼
React Frontend (Vite, React Router, ConversationContext)
  │
  ▼
Session Creation (POST /api/session/start -> SessionService -> DB: sessions table)
  │
  ▼
Narrative Input (User enters plain text narrative)
  │
  ▼
FastAPI Backend (POST /api/chat/message)
  │
  ▼
Validation (FastAPI Pydantic schema validation: session_id & message string checks)
  │
  ▼
Message Persistence (ConversationMessage & Session.messages JSON stored in PostgreSQL)
  │
  ▼
BERT Classification (predict_bert.py: AutoModelForSequenceClassification -> 6 logits -> Sigmoid -> Per-label thresholds)
  │
  ▼
Prediction Persistence (Prediction model records & Session.predicted_labels JSON updated in DB)
  │
  ▼
NER Entity Extraction (predict_ner.py: AutoModelForTokenClassification -> Argmax BIO decoding -> Character offset mapping)
  │
  ▼
Entity Persistence (Entity model records & Session.entities JSON updated in DB)
  │
  ▼
Question Engine (question_engine/engine.py: process_submission -> Rules & Known/Missing info evaluation)
  │
  ├─────────────────────────────────────────────────┐
  ▼                                                 ▼
[Missing Info Exists]                     [All Info Complete]
  │                                                 │
Follow-up Question Selected                         ▼
  │                                       Status -> PROCESSING
Session.conversation_status -> QUESTIONING          │
  │                                       Report Generation (generate_incident_report)
User Answers Question                               │
  │                                       Legal Mapping (legal_mapping.py -> DB query: laws)
Context Update (narrative + answers merged)        │
  │                                       Support Mapping (support_mapping.py -> DB query: support_services)
  └─────────────────►[Re-evaluate Engine]           │
                                          Anti-Hallucination Verification (anti_hallucination.py)
                                                    │
                                          Report Assembly (10 Conceptual Sections compiled)
                                                    │
                                          Report Persistence (Report ORM -> DB: reports table)
                                                    │
                                          Session.conversation_status -> COMPLETED
                                                    │
                                          Report Retrieval (GET /api/report/{submission_id})
                                                    │
                                          React Report Page Rendering (Report.jsx)
                                                    │
                                          PDF Generation (jsPDF client-side rendering with ABHERA_COLORS)
```

---

# PART 3 — COMPLETE END-TO-END TABLE

| Stage # | Stage Name | Input | Operation | Output | Stored Where | Next Consumer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Session Creation | `user_id` (optional), `initial_narrative` | Instantiates unique `session_id` (`SESS-xxxx`) & `submission_id` (`SUB-xxxx`) | `SessionResponse` JSON object | Memory & PostgreSQL `sessions`, `incident_submissions` | React Frontend (`ConversationContext`) |
| **2** | Narrative Input | Plain text narrative string | User types narrative into `ChatInput.jsx` UI | Raw text payload | React state (`messages`) | Axios API Client (`sendChatMessage`) |
| **3** | Input Validation | HTTP POST JSON payload | FastAPI Pydantic validator checks non-empty `session_id` & `message` | Cleaned text string | Transient server memory | `ChatService.process_message()` |
| **4** | Message Persistence | `session_id`, `user_message_text` | Appends message dict to `session.messages` list | Updated DB record | PostgreSQL `sessions.messages` JSON column | Multi-turn dialogue history |
| **5** | BERT Preprocessing | Narrative text string | Casts text to string, passes to `bert-base-uncased` tokenizer | PyTorch `input_ids`, `attention_mask` tensors | Model memory GPU/CPU | BERT Classifier forward pass |
| **6** | BERT Classification | `input_ids`, `attention_mask` | Forward pass through fine-tuned BERT -> 6 logits -> Sigmoid -> Decision thresholding | Predicted category codes (e.g. `["DV", "CA"]`) | Model memory & return dict | Prediction persistence & Question Engine |
| **7** | Prediction Persistence | Predicted label list & scores | Formats `Prediction` ORM objects & updates submission | Updated DB records | PostgreSQL `predictions` & `sessions.predicted_labels` | Chat API response & Report Generator |
| **8** | NER Token Extraction | User message text string | Token classification through fine-tuned BERT-NER -> BIO argmax decoding -> offset alignment | List of entity dicts (label, text, start, end) | Model memory & return dict | Entity persistence & Question Engine |
| **9** | Entity Persistence | Extracted entity dict list | Formats `Entity` ORM objects & updates submission | Updated DB records | PostgreSQL `entities` & `sessions.entities` | Chat API response & Report Generator |
| **10** | Question Selection | `submission_id`, categories, known entities | Evaluates required vs known info, selects highest priority unasked question | `QuestionResponse` dict with `next_question` | PostgreSQL `submission_question_answers` | React Frontend (`Chatbot.jsx`) |
| **11** | Question Answer Persistence | User answer string, `question_id` | `record_answer()` updates answer text and timestamp | `SubmissionQuestionAnswer` ORM record | PostgreSQL `submission_question_answers` & `sessions.answers` | Context updater & Question Engine |
| **12** | Context Update | Initial narrative + list of answer strings | Concatenates initial text with user answers for full narrative representation | `full_narrative` string | Transient server memory | BERT & NER re-evaluation |
| **13** | Legal Mapping | Predicted category codes (`DV`, `SH`, etc.) | Normalizes codes, executes SQL query `Law.applicable_label.in_(codes)` | List of legal provision dicts (act, section, text) | Transient server memory | Anti-Hallucination Service |
| **14** | Support Mapping | Category codes, `state`, `district` | Queries `SupportService` table, ranks by location (district=300, state=200, national=100) | Ranked list of helpline dicts | Transient server memory | Anti-Hallucination Service |
| **15** | Anti-Hallucination | Mapped legal & support records | Verifies act/section/text and service details match PostgreSQL tables verbatim | Gated payload with audit metrics | Transient server memory | Report Generator |
| **16** | Report Generation | Submission record, entities, answers, legal & support data | Assembles all 10 required conceptual report sections into JSON schema | Structured report dictionary | Transient server memory | Report Persistence |
| **17** | Report Persistence | `submission_id`, report JSON dict | Instantiates/updates `Report` ORM entity | Database record | PostgreSQL `reports.report_data` JSON column | Report API endpoint |
| **18** | Report Retrieval | `submission_id` URL path param | GET `/api/report/{submission_id}` fetches stored JSON from `reports` table | `ReportResponse` JSON | Transient server memory | React Frontend (`Report.jsx`) |
| **19** | Frontend Rendering | Report JSON object | Normalizes schema, populates cards, badges, and section UI components | Rendered DOM elements | Web browser DOM | User display & PDF trigger |
| **20** | PDF Generation | Rendered report data & DOM | `jsPDF` client-side canvas rendering using `ABHERA_COLORS` tokens | Downloadable `.pdf` binary file | Local user disk storage | End survivor user |

---

# PART 4 — DATASET

### Dataset Filename
`data/womens_safety_dataset.csv` (Split artifacts: `ml/preprocessing/train_data.csv`, `val_data.csv`, `test_data.csv`).

### Number of Records
- **Total Corpus Size:** 1,086 narrative records.
- **Training Set (70%):** 760 records.
- **Validation Set (15%):** 162 records.
- **Held-Out Test Set (15%):** 164 records.

### Columns and Meanings
1. `id` / `incident_id`: Unique alphanumeric identifier string (e.g. `INC_0001`, `TEST_0050`).
2. `narrative_text`: Raw plain English survivor incident description narrative text string.
3. `DV`: Binary integer flag (`1` or `0`) for Domestic Violence.
4. `SH`: Binary integer flag (`1` or `0`) for Sexual Harassment.
5. `ST`: Binary integer flag (`1` or `0`) for Stalking.
6. `CA`: Binary integer flag (`1` or `0`) for Cyber Abuse / Cyber Harassment.
7. `WH`: Binary integer flag (`1` or `0`) for Workplace Harassment.
8. `OV`: Binary integer flag (`1` or `0`) for Other Violence / General Harassment.
9. `ipc_bns_section`: Ground-truth statutory section annotations (reference metadata column).
10. `evidence_notes`: Evidentiary tags and notes (reference metadata column).
11. `source`: Dataset origin source tag (reference metadata column).

### Target Incident Labels
The six core binary classification target columns are:

| Code | Incident Category Name | Description & Legal Context |
| :--- | :--- | :--- |
| **DV** | Domestic Violence | Physical, emotional, economic abuse or dowry harassment by spouse or relatives (IPC 498A / BNS 85, PWDVA 2005). |
| **SH** | Sexual Harassment | Unwelcome physical contact, sexual advances, or obscene gestures in public or private spaces (IPC 354 / BNS 74, 75). |
| **ST** | Stalking | Repeatedly following, monitoring, or contacting a person physically or electronically against her will (IPC 354D / BNS 78). |
| **CA** | Cyber Abuse | Online harassment, fake profile creation, leaking private images, cyberstalking (IT Act 66E, 67, 67A). |
| **WH** | Workplace Harassment | Sexual harassment, retaliation, or intimidation by colleagues/employers at work (POSH Act 2013). |
| **OV** | Other Violence | Assault, criminal intimidation, or harassment not covered under the specific categories above (IPC 506, 509 / BNS 351). |

*Note on FPM (False Promise of Marriage):* `FPM` is defined in legal/support mapping rules and question engine rules, but in the 6-dimensional classification target matrix $\mathbf{y} \in \{0,1\}^6$, `FPM` narratives map to `OV` / `DV`.

### Data Quality Audits
- **Missing Values:** 0 null values in `narrative_text` across all 1,086 records.
- **Duplicate Records:** 0 exact duplicate row records (`df.duplicated().sum() == 0`).
- **Data Leakage:** 0 cross-split exact narrative matches confirmed via MD5 hash comparison in `run_leakage_audit.py`.

### Train / Validation / Test Split Parameters
- **Split Ratio:** 70% Train (760 rows), 15% Validation (162 rows), 15% Test (164 rows).
- **Random Seed:** `42` (`np.random.seed(42)` and `df.sample(frac=1.0, random_state=42)`).
- **Stratification:** Uniform pseudo-random shuffling. Standard single-label stratification is omitted because standard `train_test_split(stratify=...)` cannot be directly applied to multi-label target vectors $\mathbf{y} \in \{0, 1\}^6$.
- **Scripts Used:** `ml/preprocessing/prepare_classification.py`, `ml/analysis/dataset_analysis.py`.

---

# PART 5 — DATA VALIDATION

------------------------------------------------------------
### CONCEPT: Data Validation Pipeline
------------------------------------------------------------

#### WHAT IS IT?
Data validation is the automated process of checking raw input records upon ingestion to ensure that required columns exist, text fields are non-empty strings, target labels contain valid binary integers (0 or 1), and no corrupted rows enter model training or API processing.

#### WHY DID WE USE IT?
ABHERA processes sensitive survivor narratives. Passing empty text, null values, or invalid target formats into neural network tokenizers or loss functions causes runtime exceptions and invalid gradient calculations during backpropagation.

#### WHERE IS IT IMPLEMENTED?
[`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L33-L154)

#### WHICH FUNCTION / CLASS?
`ClassificationDataPreparer.prepare()`

#### IMPORTANT CODE
```python
# Exact implementation in ml/preprocessing/prepare_classification.py
valid_mask = (
    df[narrative_col].notnull() & 
    (df[narrative_col].astype(str).str.strip() != "")
)
clean_df = df[valid_mask].copy()
clean_df["label_vector"] = clean_df[self.LABEL_COLUMNS].values.tolist()
```

#### LINE/BLOCK EXPLANATION
- `df[narrative_col].notnull()` checks for non-null/non-NaN narrative entries.
- `(df[narrative_col].astype(str).str.strip() != "")` ensures narrative text is not empty or whitespace-only.
- `clean_df["label_vector"]` packs the 6 binary label columns (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`) into a 6-dimensional multi-hot list per row.

#### INPUT
Raw Pandas DataFrame loaded from `data/womens_safety_dataset.csv` (1,086 rows).

#### PROCESS
Evaluates `valid_mask`, filters out invalid rows, formats multi-hot label vectors, and computes class imbalance positive weights.

#### OUTPUT
Cleaned DataFrame `clean_df` containing 1,086 valid rows and a 6D target matrix $\mathbf{Y} \in \{0, 1\}^{1086 \times 6}$.

#### STORED WHERE?
Preprocessed CSV files: `ml/preprocessing/train_data.csv`, `val_data.csv`, `test_data.csv`.

#### NEXT CONSUMER
PyTorch `WomensSafetyDataset` class in training scripts and TF-IDF `TfidfVectorizer` in baseline comparison scripts.

#### WHY IT MATTERS
Guarantees clean, deterministic dataset splits and prevents missing-data runtime failures in deep learning training loops.

#### VIVA QUESTIONS
- **Q:** How did you validate your dataset before training?
- **Q:** Did your dataset contain missing narrative rows?

#### VIVA ANSWERS
- **A:** We used `ClassificationDataPreparer.prepare()` in `prepare_classification.py` to check for non-null, non-empty narrative strings and verify binary integer target label schemas.
- **A:** Zero narrative rows were missing out of the 1,086 raw CSV records.

#### WHAT TO MEMORIZE
1. Validation script path: `ml/preprocessing/prepare_classification.py`.
2. Total valid records retained: 1,086 out of 1,086 (0 rows dropped).
3. Distinguish Data Validation (schema/null checks) from Text Preprocessing (tokenization).

---

# PART 6 — TEXT PREPROCESSING

### Implemented vs. Non-Implemented Preprocessing Audit

The table below summarizes the exact status of standard NLP text preprocessing techniques across the ABHERA repository:

| Preprocessing Technique | Implementation Status | Exact Implementation / Reasoning |
| :--- | :--- | :--- |
| **Lowercasing** | **USED (Indirect / Built-in)** | Executed internally by `bert-base-uncased` WordPiece tokenizer and baseline `TfidfVectorizer(lowercase=True)`. |
| **Punctuation Isolation** | **USED (Indirect / Built-in)** | Executed internally by `bert-base-uncased` pre-tokenizer (separates punctuation like `.` into distinct token IDs). |
| **Punctuation Removal** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Preserved. Punctuation marks provide syntactic boundary cues for Transformer self-attention. |
| **Stop-Word Removal** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Preserved. Removing stop-words destroys grammatical sentence structure required by BERT. |
| **Stemming** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. WordPiece subwords handle suffix variations natively without rule-based stemming loss. |
| **Lemmatization** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. Morphological lemmatizers (e.g. WordNet) are not used. |
| **Spelling Correction** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. No spell-check algorithms are applied prior to tokenization. |
| **URL / HTML Removal** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. Raw text formatting is retained. |
| **Emoji / Number Removal** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. Retained as raw text string characters. |
| **Unicode Normalization** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted in dataset prep. (Helper exists only inside leakage audit script). |
| **Custom Regex Cleaning** | **NOT IMPLEMENTED IN THE CURRENT PROJECT** | Omitted. Pre-tokenizer regex cleaning is not used during dataset preparation or model ingestion. |

### Where Tokenization and Normalization are Handled
- **BERT Classifier:** `transformers.AutoTokenizer.from_pretrained("bert-base-uncased")` converts raw narrative strings into WordPiece subword token IDs (`input_ids`) and binary masks (`attention_mask`), applying lowercasing, padding, and truncation.
- **TF-IDF Vectorizer:** `sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True)` lowercases text and extracts n-gram count matrices.
- **NER Tokenizer:** `AutoTokenizer` with `return_offsets_mapping=True` maps character offsets to subwords for token classification.

---

# PART 7 — FEATURE EXTRACTION

------------------------------------------------------------
### CONCEPT: Feature Extraction & Representation
------------------------------------------------------------

#### WHAT IS IT?
Feature extraction converts raw, unstructured narrative text strings into numerical representations (sparse n-gram matrices or dense contextual embedding vectors) that machine learning algorithms can compute.

#### WHY DID WE USE IT?
Machine learning models cannot process plain English text directly. They require mathematical input matrices $\mathbf{X} \in \mathbb{R}^{N \times D}$.

#### WHERE IS IT IMPLEMENTED?
- **Classical Baselines:** [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L165-L175)
- **BERT Models:** [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L224-L250)

#### WHICH FUNCTION / CLASS?
- `TfidfVectorizer` (Scikit-learn) for baselines.
- `AutoTokenizer` & BERT Transformer Encoder (`bert-base-uncased`) for deep learning.

#### IMPORTANT CODE
```python
# Baseline TF-IDF Feature Extraction in ml/evaluation/run_baseline_comparison.py
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_features=5000,
)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_val_vec   = vectorizer.transform(X_val_raw)
X_test_vec  = vectorizer.transform(X_test_raw)
```

#### LINE/BLOCK EXPLANATION
- `ngram_range=(1, 2)` extracts unigrams (single words) and bigrams (two-word phrases like `"husband beat"`).
- `sublinear_tf=True` applies logarithmic term frequency scaling $TF = 1 + \ln(tf)$ to diminish repetitive word noise.
- `min_df=2` ignores terms appearing in fewer than 2 documents across the corpus.
- `max_features=5000` bounds the vocabulary size to the top 5,000 terms by overall term frequency.
- `fit_transform` is called strictly on `X_train_raw`, preventing test-set vocabulary leakage.

#### INPUT
Raw narrative text strings.

#### PROCESS
- **TF-IDF:** Computes Term Frequency ($TF$) and Inverse Document Frequency ($IDF$), constructing a 5,000-dimensional sparse numerical matrix.
- **BERT:** Maps subwords into 768-dimensional dense contextual vectors through 12 Transformer self-attention layers.

#### OUTPUT
- Baseline: Sparse TF-IDF matrix $\mathbf{X} \in \mathbb{R}^{N \times 5000}$.
- BERT: Dense hidden tensor $\mathbf{H} \in \mathbb{R}^{B \times L \times 768}$ and pooled `[CLS]` vector $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{B \times 768}$.

#### STORED WHERE?
In RAM during model execution / training evaluation.

#### NEXT CONSUMER
LogisticRegression / LinearSVC classifiers for baselines; Linear classification head (`768 -> 6`) for BERT.

#### WHY IT MATTERS
Feature representation determines whether a model can capture domain phraseology and subtle semantic context.

#### VIVA QUESTIONS
- **Q:** How do you convert raw text into features for your baseline models?
- **Q:** What is the difference between Feature Extraction and Feature Selection?

#### VIVA ANSWERS
- **A:** Baselines use `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=5000)` fitted exclusively on the training set.
- **A:** Feature extraction converts raw text strings into numerical vectors. Feature selection evaluates numerical features against target labels to prune uninformative columns.

#### WHAT TO MEMORIZE
1. TF-IDF parameters: `ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=2`, `max_features=5000`.
2. BERT embedding dimension: 768 dimensions per token vector.
3. Feature Extraction converts text -> numbers; Feature Selection filters numbers -> subset.

---

# PART 8 — FEATURE SELECTION

------------------------------------------------------------
### CONCEPT: Feature Selection Audit & Verification
------------------------------------------------------------

#### WHAT IS IT?
Supervised feature selection evaluates extracted numerical features against target labels $\mathbf{y}$ using statistical algorithms (e.g. Chi-Square $\chi^2$, Mutual Information, ANOVA F-test, Recursive Feature Elimination) to prune irrelevant columns.

#### REPOSITORY AUDIT RESULT
**NO EXPLICIT SUPERVISED FEATURE SELECTION ALGORITHM IS IMPLEMENTED IN THE CURRENT PROJECT.**

A complete scan across all repository files (`ml/`, `backend/`, `data/`) confirmed zero instances of `SelectKBest`, `chi2`, `mutual_info`, `RFE`, `RFECV`, `VarianceThreshold`, `PCA`, or `TruncatedSVD`.

#### WHERE IS IT DISCUSSED IN CODE?
[`ml/evaluation/ACTUAL_FEATURE_SELECTION_ANALYSIS.md`](file:///d:/Abhera(Mini)/ml/evaluation/ACTUAL_FEATURE_SELECTION_ANALYSIS.md#L168-L178)

#### EXPLANATION OF TF-IDF PARAMETERS
The parameters `max_features=5000` and `min_df=2` in `TfidfVectorizer` represent **unsupervised feature-space bounding / vocabulary limitation based on document frequency**, NOT supervised target-conditioned feature selection. They select terms based on raw text occurrences without evaluating target label correlations.

#### BERT FEATURE SELECTION
BERT does not use explicit feature selection. The Transformer encoder dynamically learns 768-dimensional class-discriminative contextual features via multi-head self-attention during backpropagation.

#### VIVA QUESTIONS
- **Q:** Did you use SelectKBest or Chi-Square feature selection in your project?
- **Q:** Is `max_features=5000` in TF-IDF considered supervised feature selection?

#### VIVA ANSWERS
- **A:** **No.** No supervised feature selection algorithms (like SelectKBest, Chi-Square, or RFE) are implemented in the project. Baselines use all 5,000 extracted TF-IDF features directly, and BERT uses learned self-attention embeddings.
- **A:** **No.** `max_features=5000` is an unsupervised vocabulary constraint based purely on term frequency across raw narratives, independent of target labels.

#### WHAT TO MEMORIZE
1. Explicit feature selection status: **NOT IMPLEMENTED IN CURRENT PROJECT**.
2. Baseline feature dimension passed to classifiers: All 5,000 TF-IDF features.
3. BERT feature selection: Learned end-to-end via Transformer self-attention.

---

# PART 9 — CLASSICAL BASELINES

------------------------------------------------------------
### CONCEPT: Classical Baseline Machine Learning Models
------------------------------------------------------------

#### WHAT IS IT?
Classical baseline models are traditional machine learning algorithms (Logistic Regression and Linear Support Vector Machines) trained on sparse TF-IDF n-gram matrices to establish a benchmark for evaluating deep learning transformer performance.

#### WHY DID WE USE IT?
To scientifically prove whether fine-tuning a deep BERT transformer provides statistically significant performance gains over standard linear NLP approaches.

#### WHERE IS IT IMPLEMENTED?
- [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L180-L205)
- [`ml/evaluation/baseline_logistic_regression.py`](file:///d:/Abhera(Mini)/ml/evaluation/baseline_logistic_regression.py)
- [`ml/evaluation/baseline_linear_svm.py`](file:///d:/Abhera(Mini)/ml/evaluation/baseline_linear_svm.py)

#### WHICH FUNCTION / CLASS?
`OneVsRestClassifier(LogisticRegression(...))` and `OneVsRestClassifier(LinearSVC(...))` from Scikit-Learn.

#### IMPORTANT CODE
```python
# Baseline Classifier setup in ml/evaluation/run_baseline_comparison.py
log_reg = OneVsRestClassifier(
    LogisticRegression(C=1.0, max_iter=1000, random_state=42)
)
log_reg.fit(X_train_vec, Y_train)

svm_model = OneVsRestClassifier(
    LinearSVC(C=1.0, max_iter=2000, random_state=42)
)
svm_model.fit(X_train_vec, Y_train)
```

#### LINE/BLOCK EXPLANATION
- `OneVsRestClassifier` fits 6 independent binary linear classifiers (one per incident category $k \in \{1 \dots 6\}$).
- `LogisticRegression(C=1.0)` uses L2 regularization to fit log-odds decision boundaries.
- `LinearSVC(C=1.0)` fits maximum-margin hyperplanes using linear support vector classification.

#### EXPERIMENTAL BASELINE RESULTS (TEST SET, N=164)

| Baseline Model | Exact Match Accuracy | Micro Precision | Micro Recall | Micro F1-Score | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 95.73% (0.9573) | 97.79% (0.9779) | 97.79% (0.9779) | **97.79% (0.9779)** | 97.35% (0.9735) |
| **TF-IDF + Linear SVM** | 95.12% (0.9512) | 97.83% (0.9783) | 96.72% (0.9672) | **97.27% (0.9727)** | 96.69% (0.9669) |

#### WHY BASELINES PERFORM STRONGLY
Survivor incident narratives contain strong, highly explicit lexical keywords (e.g., `"husband"`, `"slapped"`, `"dowry"`, `"office manager"`, `"instagram fake profile"`). TF-IDF n-grams capture these overt lexical cues effectively.

#### VIVA QUESTIONS
- **Q:** What classical baseline models did you train?
- **Q:** How did you handle multi-label classification for Logistic Regression and SVM?

#### VIVA ANSWERS
- **A:** We trained TF-IDF + Logistic Regression and TF-IDF + Linear SVM using a 5,000-dimensional unigram/bigram feature space.
- **A:** We used a `OneVsRestClassifier` meta-estimator, fitting 6 independent binary linear classifiers for the 6 incident labels.

#### WHAT TO MEMORIZE
1. Logistic Regression Micro-F1: 97.79%.
2. Linear SVM Micro-F1: 97.27%.
3. Multi-label strategy used for baselines: `OneVsRestClassifier`.

---

# PART 10 — BERT CONCEPT

------------------------------------------------------------
### CONCEPT: Bidirectional Encoder Representations from Transformers (BERT)
------------------------------------------------------------

#### WHAT IS IT?
BERT is a deep learning transformer architecture developed by Google. Unlike traditional unidirectional language models that read text left-to-right or right-to-left, BERT reads sequence tokens bidirectionally across all self-attention layers simultaneously, capturing deep contextual relationships between words.

#### WHY DID WE USE IT?
Harassment narratives contain subtle contextual nuances where word meaning depends heavily on surrounding context (e.g., distinguishing between a supervisor giving work instructions vs. making coercive threats). Pretrained BERT captures rich semantic context without requiring manual feature engineering.

#### PRETRAINED BASE MODEL USED
**`bert-base-uncased`**
- **Parameters:** 110 Million.
- **Transformer Layers (Encoder Blocks):** 12.
- **Hidden Layer Dimension ($d_{\text{model}}$):** 768.
- **Self-Attention Heads:** 12.
- **Vocabulary Size:** 30,522 WordPiece tokens.
- **"Uncased" Meaning:** All text is converted to lowercase prior to subword tokenization, treating `"Harassment"` and `"harassment"` identically.

#### PRETRAINING VS. FINE-TUNING
- **Pretraining:** Training BERT on massive unannotated text corpora (BookCorpus and English Wikipedia) using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks to learn general English grammar and semantic representations.
- **Fine-Tuning:** Updating pretrained weights by training BERT end-to-end on ABHERA's survivor dataset (`womens_safety_dataset.csv`) to perform multi-label harassment classification and named entity recognition.

#### VIVA QUESTIONS
- **Q:** What pretrained BERT model did you fine-tune?
- **Q:** What does "uncased" mean in `bert-base-uncased`?
- **Q:** Why use pretrained BERT instead of training a transformer from scratch?

#### VIVA ANSWERS
- **A:** We fine-tuned `bert-base-uncased` (110M parameters, 12 layers, 768 hidden size).
- **A:** "Uncased" means all text is lowercased during tokenization, ignoring capitalization differences.
- **A:** Pretrained BERT already understands English syntax and semantics from billions of words, allowing us to achieve high accuracy on a domain-specific dataset with lightweight fine-tuning.

#### WHAT TO MEMORIZE
1. Base model name: `bert-base-uncased`.
2. Model dimensions: 12 layers, 768 hidden size, 12 attention heads, 110M parameters.
3. Pretraining (general language learning) vs Fine-Tuning (domain task adaptation).

---

# PART 11 — BERT INPUT PROCESSING

------------------------------------------------------------
### CONCEPT: BERT Tokenization and Input Tensors
------------------------------------------------------------

#### WHAT IS IT?
BERT input processing converts raw narrative text strings into standardized integer tensors (`input_ids` and `attention_mask`) required by the Transformer encoder.

#### WHERE IS IT IMPLEMENTED?
- Training: [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L46-L52)
- Production Inference: [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L248-L260)

#### TOKENIZATION PIPELINE FLOW
```
Raw Narrative String
        │
        ▼
Lowercasing & WordPiece Subword Tokenization
        │
        ▼
Special Token Insertion ([CLS] at start, [SEP] at end)
        │
        ▼
Padding / Truncation to Fixed Maximum Sequence Length
        │
        ▼
PyTorch Input Tensors (input_ids & attention_mask)
```

#### SPECIAL TOKENS DEFINED
- **`[CLS]` (Classification Token):** Token ID `101`. Inserted at index 0 of every input sequence. Its final hidden state $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$ aggregates whole-sequence context for classification.
- **`[SEP]` (Separator Token):** Token ID `102`. Inserted at the end of the narrative sequence to mark the sentence boundary.
- **`[PAD]` (Padding Token):** Token ID `0`. Appended to pad shorter sequences to the uniform `max_length`.

#### SEQUENCE LENGTH SPECIFICATIONS
- **BERT v1 Training:** `max_length = 64`.
- **BERT v2 Training (Promoted):** `max_length = 128`.
- **Validation & Test Evaluation:** `max_length = 128` (eval) / `256` (full narrative inference).
- **Production Inference (`predict_bert.py`):** `max_length = 256`.

#### ATTENTION MASK
Binary tensor ($\text{shape} = [B, L]$) containing `1` for real tokens (`[CLS]`, subwords, `[SEP]`) and `0` for padded tokens (`[PAD]`). Prevents self-attention layers from computing attention over pad tokens.

#### VIVA QUESTIONS
- **Q:** What special tokens does BERT use and why?
- **Q:** What sequence length was used in production inference?
- **Q:** What is the role of the `attention_mask`?

#### VIVA ANSWERS
- **A:** `[CLS]` (token ID 101) at index 0 for sequence classification, `[SEP]` (102) to mark sequence end, and `[PAD]` (0) for zero-padding.
- **A:** `max_length = 256` tokens in `predict_bert.py` to prevent text truncation during multi-turn intake.
- **A:** The `attention_mask` tells self-attention layers to ignore `[PAD]` tokens (`0`) and only attend to valid text tokens (`1`).

#### WHAT TO MEMORIZE
1. `[CLS]` = Token 101, `[SEP]` = Token 102, `[PAD]` = Token 0.
2. Max sequence length: 128 during v2 training, 256 during production inference.
3. `attention_mask` contains 1 for real tokens, 0 for padded tokens.

---

# PART 12 — BERT ARCHITECTURE

------------------------------------------------------------
### CONCEPT: BERT Classification Architecture & Network Forward Pass
------------------------------------------------------------

#### WHAT IS IT?
The classification architecture consists of the fine-tuned 12-layer `bert-base-uncased` Transformer encoder connected to a linear multi-label classification head that outputs 6 independent class probabilities.

#### ARCHITECTURE PIPELINE FLOW
```
Input Tensors (input_ids, attention_mask)  [Batch Size B x Sequence Length L]
                     │
                     ▼
       BERT Transformer Encoder (12 Layers)
                     │
                     ▼
   Hidden States Matrix H  [B x L x 768]
                     │
                     ▼
   [CLS] Token Extraction (Index 0)  [B x 768]
                     │
                     ▼
      Dropout Layer (probability = 0.1)
                     │
                     ▼
 Linear Classification Head (W: 768 -> 6, b: 6)
                     │
                     ▼
         6 Output Logits z_k  [B x 6]
                     │
                     ▼
          Elementwise Sigmoids p_k = σ(z_k)
                     │
                     ▼
     Label-Specific Thresholding (y_k = 1 if p_k >= τ_k)
```

#### MATHEMATICAL FORMULATION
1. **[CLS] Representation Extraction:**  
   $$\mathbf{h}_{\text{CLS}} = \mathbf{H}[:, 0, :] \quad \in \mathbb{R}^{B \times 768}$$
2. **Classification Head Logit Computation:**  
   $$\mathbf{z} = \mathbf{h}_{\text{CLS}} \mathbf{W}^T + \mathbf{b} \quad \in \mathbb{R}^{B \times 6}$$
3. **Independent Sigmoid Probability Computation:**  
   $$p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \quad \text{for } k \in \{1 \dots 6\}$$
4. **Binary Label Decision Rule:**  
   $$\hat{y}_k = \begin{cases} 1 & \text{if } p_k \ge \tau_k \\ 0 & \text{otherwise} \end{cases}$$

#### VIVA QUESTIONS
- **Q:** How does BERT generate predictions for multi-label classification?
- **Q:** What vector is passed to the final linear layer?

#### VIVA ANSWERS
- **A:** The 768-dimensional `[CLS]` token output vector $\mathbf{h}_{\text{CLS}}$ is passed to a linear layer (`768 -> 6`) to compute 6 logits. Sigmoid activation then converts logits into 6 independent probabilities.
- **A:** The `[CLS]` token vector at index 0 of the final hidden layer.

#### WHAT TO MEMORIZE
1. Hidden state dimension: 768.
2. Linear head mapping: `Linear(in_features=768, out_features=6)`.
3. Output activation: Sigmoid $\sigma(z_k)$ producing 6 independent probabilities $p_k \in [0, 1]$.

---

# PART 13 — MULTI-LABEL CLASSIFICATION

------------------------------------------------------------
### CONCEPT: Multi-Label vs. Multi-Class Classification
------------------------------------------------------------

#### WHAT IS IT?
Multi-label classification allows a single incident narrative to belong to zero, one, or multiple target categories simultaneously (e.g. `[DV=1, CA=1, SH=0, ST=0, WH=0, OV=0]`).

#### WHY DID WE USE IT?
Real-world survivor incidents are frequently compound incidents. For example, a perpetrator may commit physical domestic abuse at home while simultaneously sending extortion messages online (`DV` + `CA`). A single-class classifier would force an incorrect choice between the two.

#### MULTI-CLASS VS MULTI-LABEL COMPARISON

| Dimension | Multi-Class Classification | Multi-Label Classification (ABHERA) |
| :--- | :--- | :--- |
| **Class Assignment** | Exactly 1 class per sample. | 0, 1, or multiple classes per sample. |
| **Classes Sum Constraint** | Probabilities sum to 1.0 ($\sum p_k = 1.0$). | Probabilities are mutually independent ($p_k \in [0, 1]$). |
| **Output Activation** | **Softmax** $\frac{e^{z_k}}{\sum e^{z_j}}$ | **Sigmoid** $\frac{1}{1 + e^{-z_k}}$ |
| **Loss Function** | Categorical Cross-Entropy | Binary Cross-Entropy (`BCEWithLogitsLoss`) |
| **Target Matrix** | One-hot vector $\mathbf{y} \in \{0, 1\}^K$ ($\sum y_k = 1$) | Multi-hot vector $\mathbf{y} \in \{0, 1\}^K$ |

#### 6-DIMENSIONAL TARGET VECTOR
Target vector order is strictly fixed:  
$$\mathbf{y} = [y_{\text{DV}}, y_{\text{SH}}, y_{\text{ST}}, y_{\text{CA}}, y_{\text{WH}}, y_{\text{OV}}]^T \in \{0, 1\}^6$$

Example compound incident:  
Narrative: *"My husband beat me at home and posted my private photos online."*  
$$\mathbf{y} = [1, 0, 0, 1, 0, 0]^T \quad (\text{DV}=1, \text{CA}=1)$$

#### VIVA QUESTIONS
- **Q:** Why did you use multi-label classification instead of multi-class classification?
- **Q:** Why use Sigmoid instead of Softmax?

#### VIVA ANSWERS
- **A:** Harassment incidents are often compound events (e.g. Domestic Violence + Cyber Abuse). Multi-label classification allows predicting multiple non-mutually exclusive categories for a single narrative.
- **A:** Softmax forces class probabilities to sum to 1.0, making classes compete. Sigmoid computes independent probabilities for each category.

#### WHAT TO MEMORIZE
1. Multi-class = Softmax (sums to 1); Multi-label = Sigmoid (independent probabilities).
2. ABHERA target vector dimension: 6-dimensional binary multi-hot vector.
3. Label order: `[DV, SH, ST, CA, WH, OV]`.

---

# PART 14 — LOSS FUNCTION

------------------------------------------------------------
### CONCEPT: BCEWithLogitsLoss and Positive-Class Weighting
------------------------------------------------------------

#### WHAT IS IT?
`BCEWithLogitsLoss` combines a Sigmoid layer and Binary Cross-Entropy loss in a single numerically stable PyTorch class, incorporating positive-class weighting (`pos_weight`) to penalize false negatives on imbalanced datasets.

#### WHERE IS IT IMPLEMENTED?
[`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L274-L285)

#### IMPORTANT CODE
```python
# Loss Function setup in ml/training/train_bert_experiment_v2.py
pos_weights = torch.tensor([
    pos_weight_dict[col] for col in label_columns
], dtype=torch.float).to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

#### MATHEMATICAL FORMULATION
For a batch of $B$ samples and $K=6$ classes, the weighted Binary Cross-Entropy loss $\mathcal{L}$ is computed as:

$$\mathcal{L} = -\frac{1}{B \cdot K} \sum_{i=1}^{B} \sum_{k=1}^{K} \left[ w_k \cdot y_{i,k} \cdot \ln(\sigma(z_{i,k})) + (1 - y_{i,k}) \cdot \ln(1 - \sigma(z_{i,k})) \right]$$

Where:
- $z_{i,k}$ is the raw logit output for sample $i$ and class $k$.
- $\sigma(z_{i,k}) = \frac{1}{1 + e^{-z_{i,k}}}$ is the sigmoid probability.
- $y_{i,k} \in \{0, 1\}$ is the ground-truth binary label.
- $w_k = \frac{N_{\text{total}} - N_{\text{pos}, k}}{N_{\text{pos}, k}}$ is the positive class weight for category $k$.

#### CLASS IMBALANCE & POS_WEIGHT RATIONALE
In safety datasets, negative samples ($y_k = 0$) outnumber positive samples ($y_k = 1$). Without positive weighting, standard BCE causes the model to predict all zeros to minimize loss, producing high false negatives. Setting $w_k > 1$ multiplies the loss penalty whenever the model misses a true positive incident ($y_k = 1$).

#### VIVA QUESTIONS
- **Q:** What loss function did you use for training BERT?
- **Q:** What is `pos_weight` and why was it necessary?

#### VIVA ANSWERS
- **A:** We used `torch.nn.BCEWithLogitsLoss` with positive-class weighting (`pos_weight`).
- **A:** `pos_weight` assigns higher loss penalties to false negative predictions on positive incident samples, mitigating class imbalance and reducing false negatives.

#### WHAT TO MEMORIZE
1. Loss function name: `BCEWithLogitsLoss`.
2. Positive weight formula: $w_k = (N_{\text{total}} - N_{\text{pos}, k}) / N_{\text{pos}, k}$.
3. Numerical stability benefit: Combines Sigmoid + BCE to prevent log-underflow/overflow.

---

# PART 15 — TRAINING

------------------------------------------------------------
### CONCEPT: BERT Fine-Tuning Hyperparameters & Training Loop
------------------------------------------------------------

#### WHAT IS IT?
The complete end-to-end training procedure used to adapt `bert-base-uncased` weights on the 760 training samples of the ABHERA dataset.

#### WHERE IS IT IMPLEMENTED?
[`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L220-L310)

#### PRODUCTION TRAINING HYPERPARAMETERS

| Hyperparameter | Value | Rationale / Explanation |
| :--- | :--- | :--- |
| **Base Model** | `bert-base-uncased` | Pretrained 12-layer Transformer encoder (110M parameters). |
| **Epochs** | `6` | Provides sufficient optimization steps over 760 training samples. |
| **Batch Size** | `32` | Standard batch size balancing gradient estimate variance and GPU memory. |
| **Learning Rate** | `3e-5` | Standard AdamW learning rate for stable transformer fine-tuning. |
| **Weight Decay** | `0.01` | L2 regularization applied to non-bias/layer-norm parameters to prevent overfitting. |
| **Optimizer** | `AdamW` | Adaptive Moment Estimation with decoupled weight decay. |
| **Warmup Ratio** | `0.10` (10%) | Linearly ramps learning rate over initial 10% of training steps. |
| **Max Sequence Length** | `128` | Captures narrative context without GPU memory saturation. |
| **Dropout** | `0.10` | 10% dropout applied before the linear classification head. |
| **Random Seed** | `42` | Ensures complete experimental reproducibility (`torch.manual_seed(42)`). |

#### TRAINING LOOP STEP-BY-STEP
1. **Batch Ingestion:** Load batch of narratives and 6D target vectors ($B=32$).
2. **Tokenization:** Convert raw narratives to `input_ids` and `attention_mask` tensors.
3. **Forward Pass:** Compute logits $\mathbf{z} \in \mathbb{R}^{32 \times 6}$ through BERT.
4. **Loss Computation:** Compute weighted loss $\mathcal{L} = \text{BCEWithLogitsLoss}(\mathbf{z}, \mathbf{y}, \text{pos\_weight})$.
5. **Backpropagation:** Execute `loss.backward()` to compute gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$.
6. **Gradient Clipping:** Clip gradient norm `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
7. **Optimizer Step:** Update model weights via `optimizer.step()` and `scheduler.step()`.
8. **Zero Gradients:** Reset gradients via `optimizer.zero_grad()`.

#### VIVA QUESTIONS
- **Q:** What optimizer and learning rate did you use?
- **Q:** How many epochs did you train BERT v2 for?
- **Q:** What is AdamW and weight decay?

#### VIVA ANSWERS
- **A:** We used the AdamW optimizer with a learning rate of $3 \times 10^{-5}$ and 10% linear warmup.
- **A:** We trained BERT v2 for 6 epochs with a batch size of 32.
- **A:** AdamW is Adam with decoupled weight decay ($0.01$), which penalizes large weights to prevent overfitting.

#### WHAT TO MEMORIZE
1. Epochs: 6; Batch size: 32; Learning rate: 3e-5.
2. Optimizer: AdamW; Weight decay: 0.01; Warmup: 10%.
3. Gradient clipping max norm: 1.0.

---

# PART 16 — BERT V1

------------------------------------------------------------
### CONCEPT: BERT v1 Baseline Model & Architecture Limitations
------------------------------------------------------------

#### WHAT IS IT?
BERT v1 represents the initial baseline fine-tuned model trained prior to hyperparameter optimization, threshold tuning, and weighted loss integration.

#### WHERE IS IT IMPLEMENTED / STORED?
- Artifact Directory: `models/bert_multilabel/`
- Configuration File: `models/bert_multilabel/training_config.json`
- Evaluation Results: `models/bert_multilabel/evaluation_results.json`

#### BERT V1 CONFIGURATION
- **Epochs:** 2.
- **Max Sequence Length:** 64.
- **Batch Size:** 32.
- **Learning Rate:** 3e-5.
- **Loss Function:** Standard unweighted `BCEWithLogitsLoss`.
- **Decision Threshold:** Fixed global threshold $\tau = 0.50$ across all 6 labels.

#### BERT V1 TEST SET PERFORMANCE (N=164)
- **Exact Match Accuracy:** 83.54% (0.8354).
- **Micro Precision:** 95.76% (0.9576).
- **Micro Recall:** 87.29% (0.8729).
- **Micro F1-Score:** 91.33% (0.9133).
- **Macro F1-Score:** 90.37% (0.9037).
- **Multi-Label Exact Match:** **0.00% (0.0000)** (0 / 17 multi-label test cases correct).
- **Total False Negatives:** **23**.
- **Total False Positives:** 7.

#### VIVA QUESTIONS
- **Q:** What were the main limitations of BERT v1?
- **Q:** Why did BERT v1 fail on multi-label test cases?

#### VIVA ANSWERS
- **A:** BERT v1 suffered from under-training (only 2 epochs), sequence truncation (`max_length=64`), unweighted BCE loss, and a rigid global 0.50 decision threshold.
- **A:** Due to sequence truncation and rigid 0.50 thresholds, BERT v1 missed secondary incident labels in compound narratives, yielding 23 false negatives and 0% multi-label exact match.

#### WHAT TO MEMORIZE
1. BERT v1 Micro-F1: 91.33%; Exact Match: 83.54%.
2. BERT v1 False Negatives: 23; Multi-label Exact Match: 0.0%.
3. Key flaws: Under-training (2 epochs), short sequence cutoff (64), default 0.50 thresholds.

---

# PART 17 — BERT V1 ERROR ANALYSIS

------------------------------------------------------------
### CONCEPT: BERT v1 Systematic Error Analysis
------------------------------------------------------------

#### WHAT IS IT?
A diagnostic investigation analyzing the 30 total prediction errors (23 false negatives, 7 false positives) produced by BERT v1 on the held-out test set.

#### WHERE IS IT DOCUMENTED?
- [`ml/evaluation/bert_error_analysis_report.md`](file:///d:/Abhera(Mini)/ml/evaluation/bert_error_analysis_report.md)
- `ml/evaluation/bert_error_analysis.json`
- `ml/evaluation/bert_error_analysis.csv`

#### IDENTIFIED ROOT CAUSES OF BERT V1 ERRORS

1. **Under-Training (Insufficient Epochs):** Training for only 2 epochs left the classification head parameters under-optimized, causing predicted probabilities for active positive labels to hover around 0.35–0.48 (just below the default 0.50 decision threshold).
2. **Sequence Truncation Noise (`max_length=64`):** Longer survivor narratives were truncated at 64 tokens. Secondary incident mentions appearing later in narrative text (e.g. cyber harassment threats mentioned at the end of a domestic abuse story) were cut off before reaching the Transformer encoder.
3. **Class Imbalance & Loss Attenuation:** Unweighted BCE loss allowed the overwhelming majority of negative binary instances ($y_k = 0$) to dominate gradients, pulling positive label probabilities down.
4. **Rigid Global Threshold ($\tau = 0.50$):** Using a uniform 0.50 cutoff failed to account for category-specific probability distributions (e.g. Workplace Harassment probabilities rarely exceeded 0.35 in compound narratives).

#### VIVA QUESTIONS
- **Q:** How many false negatives did BERT v1 make and why?
- **Q:** How did error analysis guide the development of BERT v2?

#### VIVA ANSWERS
- **A:** BERT v1 produced 23 false negatives due to sequence truncation (`max_length=64`), under-training (2 epochs), and unweighted BCE loss.
- **A:** Error analysis demonstrated the need to increase sequence length to 128/256, extend training to 6 epochs, apply positive loss weighting, and perform per-label threshold tuning.

#### WHAT TO MEMORIZE
1. Primary error mode in v1: False negatives (23 FN vs 7 FP).
2. Main technical cause: Sequence truncation at 64 tokens + unweighted BCE.

---

# PART 18 — BERT V2

------------------------------------------------------------
### CONCEPT: BERT v2 Promoted Production Model Improvements
------------------------------------------------------------

#### WHAT IS IT?
BERT v2 is the audited and promoted production classification model developed to resolve the structural flaws identified during BERT v1 error analysis.

#### WHERE IS IT IMPLEMENTED / STORED?
- Production Directory: `models/bert_multilabel_experiment_v2/`
- Training Script: [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py)
- Promotion Documentation: [`ml/evaluation/bert_v2_production_promotion.md`](file:///d:/Abhera(Mini)/ml/evaluation/bert_v2_production_promotion.md)

#### STRUCTURAL COMPARISON: BERT V1 VS. BERT V2

| Technical Dimension | BERT v1 (Baseline) | BERT v2 (Promoted Production) | Rationale for Change |
| :--- | :--- | :--- | :--- |
| **Training Epochs** | 2 Epochs | **6 Epochs** | Eliminates under-training; ensures converged loss. |
| **Max Sequence Length** | 64 Tokens | **128 (Train) / 256 (Inference)** | Prevents text truncation; preserves full narrative context. |
| **Loss Function** | Standard Unweighted BCE | **BCEWithLogitsLoss + `pos_weight`** | Penalizes false negatives on positive class instances. |
| **Optimizer & Decay** | AdamW (lr=3e-5) | **AdamW (lr=3e-5, weight_decay=0.01)** | Regularizes weights; prevents overfitting over 6 epochs. |
| **Decision Thresholds** | Fixed Global $\tau = 0.50$ | **6 Label-Specific Thresholds ($\tau_k$)** | Optimizes per-label F1 scores on validation data. |

#### VIVA QUESTIONS
- **Q:** What exact changes were made in BERT v2?
- **Q:** Did BERT v2 reduce false negatives?

#### VIVA ANSWERS
- **A:** Increased epochs from 2 to 6, increased max sequence length from 64 to 128/256, added positive-class loss weighting (`pos_weight`), added weight decay ($0.01$), and tuned label-specific decision thresholds.
- **A:** Yes. False negatives dropped from 23 in v1 down to 1 in v2—a **95.65% reduction in false negative errors**.

#### WHAT TO MEMORIZE
1. Epochs: 2 -> 6; Max length: 64 -> 128/256.
2. Loss: Unweighted -> `pos_weight` weighted BCE.
3. Thresholds: Global 0.50 -> 6 validation-tuned thresholds.

---

# PART 19 — THRESHOLD TUNING

------------------------------------------------------------
### CONCEPT: Validation-Only Per-Label Decision Threshold Optimization
------------------------------------------------------------

#### WHAT IS IT?
The process of searching and selecting optimal decision thresholds $\tau_k \in [0.10, 0.90]$ for each of the 6 incident categories independently to maximize per-label F1 scores.

#### WHY DID WE USE IT?
In multi-label classification with class imbalance, a default threshold of $0.50$ is rarely optimal. Low-frequency categories (e.g. Workplace Harassment) may generate strong semantic confidence at $p=0.30$, while frequent categories (e.g. Stalking) require $p=0.75$ to filter out false positives.

#### WHERE IS IT IMPLEMENTED?
- Script: [`ml/evaluation/verify_optimal_thresholds.py`](file:///d:/Abhera(Mini)/ml/evaluation/verify_optimal_thresholds.py)
- Stored Artifact: `models/bert_multilabel_experiment_v2/optimal_thresholds.json`

#### THRESHOLD TUNING METHODOLOGY (STRICT ISOLATION)
1. **Validation Search:** Grid search executed over $\tau \in [0.10, 0.90]$ at step size $0.01$ **strictly on validation probabilities ($N=162$)**.
2. **Zero Test Contamination:** Held-out test data ($N=164$) was **NEVER** exposed during threshold search.
3. **Freezing:** Optimal validation thresholds were frozen into `optimal_thresholds.json`.
4. **Evaluation:** Frozen thresholds were applied directly to test set probabilities during final evaluation.

#### FINAL OPTIMAL DECISION THRESHOLDS

| Category Code | Incident Category Name | Optimal Decision Threshold ($\tau_k$) | Rationale |
| :--- | :--- | :---: | :--- |
| **DV** | Domestic Violence | **0.73** | High threshold filters out mild domestic dispute noise. |
| **SH** | Sexual Harassment | **0.50** | Standard balanced threshold maximizes precision and recall. |
| **ST** | Stalking | **0.79** | High threshold eliminates general harassment overlap. |
| **CA** | Cyber Abuse | **0.45** | Slightly lower threshold captures subtle online threats. |
| **WH** | Workplace Harassment | **0.29** | Low threshold compensates for lower training sample frequency. |
| **OV** | Other Violence | **0.63** | Moderate-high threshold isolates general violent assault. |

#### VIVA QUESTIONS
- **Q:** How did you tune your decision thresholds?
- **Q:** Did you use the test set to find thresholds?

#### VIVA ANSWERS
- **A:** We performed per-label F1 grid search across $[0.10, 0.90]$ strictly on validation probabilities to select optimal threshold $\tau_k$ per category.
- **A:** **No.** The test set was strictly isolated. Thresholds were selected on validation data alone and frozen before test set evaluation.

#### WHAT TO MEMORIZE
1. Threshold values: DV=0.73, SH=0.50, ST=0.79, CA=0.45, WH=0.29, OV=0.63.
2. Lowest threshold: WH (0.29); Highest threshold: ST (0.79).
3. Data split used for threshold search: Validation set ($N=162$).

---

# PART 20 — FINAL BERT RESULTS

------------------------------------------------------------
### CONCEPT: Final Promoted BERT v2 Performance Evaluation
------------------------------------------------------------

#### WHAT IS IT?
The definitive, audited performance metrics of fine-tuned BERT v2 evaluated on the held-out test set ($N=164$ records, $164 \times 6 = 984$ binary classification decisions).

#### WHERE IS IT STORED?
- [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L54-L79)
- [`ml/evaluation/PAPER_RESULTS_SECTION.md`](file:///d:/Abhera(Mini)/ml/evaluation/PAPER_RESULTS_SECTION.md)

#### SUMMARY OF FINAL OVERALL METRICS

| Metric Name | Exact Mathematical Value | Percentage | Scientific Definition / Meaning |
| :--- | :---: | :---: | :--- |
| **Exact Match Accuracy** | `0.9573` | **95.73%** | Fraction of test samples where ALL 6 predicted labels match ground-truth vector exactly (157 / 164 samples). |
| **Elementwise Binary Accuracy** | `0.9929` | **99.29%** | Fraction of individual binary decisions correct across all label slots (977 / 984 binary decisions). |
| **Micro Precision** | `0.9677` | **96.77%** | Globally aggregated precision $\frac{\text{Total TP}}{\text{Total TP} + \text{Total FP}} = \frac{180}{180 + 6}$. |
| **Micro Recall** | `0.9945` | **99.45%** | Globally aggregated recall $\frac{\text{Total TP}}{\text{Total TP} + \text{Total FN}} = \frac{180}{180 + 1}$. |
| **Micro F1-Score** | `0.9809` | **98.09%** | Harmonic mean of Micro Precision and Micro Recall. |
| **Macro Precision** | `0.9571` | **95.71%** | Unweighted average of per-label precision scores. |
| **Macro Recall** | `0.9963` | **99.63%** | Unweighted average of per-label recall scores. |
| **Macro F1-Score** | `0.9758` | **97.58%** | Unweighted average of per-label F1 scores across 6 categories. |
| **Multi-Label Exact Match** | `0.9412` | **94.12%** | Exact match accuracy evaluated strictly on multi-label test cases (16 / 17 multi-label samples correct). |

#### GLOBAL CONFUSION MATRIX AGGREGATES
- **True Positives (TP):** `180`
- **False Positives (FP):** `6`
- **False Negatives (FN):** `1`
- **True Negatives (TN):** `797`
- **Total Binary Decisions:** `180 + 6 + 1 + 797 = 984`.

#### PER-LABEL PERFORMANCE BREAKDOWN TABLE

| Category Code | Threshold ($\tau_k$) | Support | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DV** | 0.73 | 35 | 35 | 2 | 0 | 127 | 94.59% | 100.00% | **97.22%** |
| **SH** | 0.50 | 45 | 44 | 0 | 1 | 119 | 100.00% | 97.78% | **98.88%** |
| **ST** | 0.79 | 32 | 32 | 0 | 0 | 132 | 100.00% | 100.00% | **100.00%** |
| **CA** | 0.45 | 39 | 39 | 1 | 0 | 124 | 97.50% | 100.00% | **98.73%** |
| **WH** | 0.29 | 19 | 19 | 2 | 0 | 143 | 90.48% | 100.00% | **95.00%** |
| **OV** | 0.63 | 11 | 11 | 1 | 0 | 152 | 91.67% | 100.00% | **95.65%** |
| **OVERALL (MICRO)** | — | **181** | **180** | **6** | **1** | **797** | **96.77%** | **99.45%** | **98.09%** |

#### SINGLE-LABEL VS MULTI-LABEL PERFORMANCE

| Subset | Total Test Samples | Exact Matches | Exact Match Accuracy |
| :--- | :---: | :---: | :---: |
| **Single-Label Incidents** | 147 | 141 | **95.92%** |
| **Multi-Label Incidents** | 17 | 16 | **94.12%** |
| **Combined Test Corpus** | **164** | **157** | **95.73%** |

#### REMAINING SINGLE FALSE NEGATIVE CASE ANALYSIS
- **Incident ID:** `TEST_0050`
- **Narrative Text:** *"An anonymous account has been sending me explicit messages for weeks, and recently the person revealed they know my home address."*
- **Ground Truth Labels:** `[SH, CA]` (Sexual Harassment, Cyber Abuse).
- **Predicted Labels:** `[CA]` (Cyber Abuse).
- **Probabilities & Thresholds:**
  - `CA`: Probability $p=0.9498 \ge 0.45$ ($\text{Threshold} = 0.45$) $\rightarrow$ **Correctly Predicted**.
  - `SH`: Probability $p=0.4549 < 0.50$ ($\text{Threshold} = 0.50$) $\rightarrow$ **Missed by marginal margin of 0.0451**.

#### VIVA QUESTIONS
- **Q:** What is your final BERT Micro F1 score?
- **Q:** What was the single false negative incident in your test evaluation?

#### VIVA ANSWERS
- **A:** Final BERT v2 achieved a **Micro F1-score of 98.09%** and **Exact Match accuracy of 95.73%**.
- **A:** Incident `TEST_0050` where Sexual Harassment probability ($0.4549$) missed its decision threshold ($0.50$) by just $0.0451$.

#### WHAT TO MEMORIZE
1. Micro F1: 98.09%; Exact Match: 95.73%; Macro F1: 97.58%.
2. Confusion matrix totals: TP=180, FP=6, FN=1, TN=797.
3. Single remaining FN case: `TEST_0050` (SH prob 0.4549 vs thresh 0.50).

---

# PART 21 — BERT V1 VS V2

------------------------------------------------------------
### CONCEPT: BERT v1 vs. BERT v2 Comparative Analysis
------------------------------------------------------------

#### WHAT IS IT?
A direct, quantitative side-by-side comparison illustrating the empirical performance improvements achieved by transitioning from baseline BERT v1 to promoted BERT v2.

#### METRIC COMPARISON TABLE

| Metric | BERT v1 (Baseline) | BERT v2 (Promoted Production) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Micro F1-Score** | 91.33% (0.9133) | **98.09% (0.9809)** | **+6.76 percentage points** |
| **Exact Match Accuracy** | 83.54% (0.8354) | **95.73% (0.9573)** | **+12.19 percentage points** |
| **Macro F1-Score** | 90.37% (0.9037) | **97.58% (0.9758)** | **+7.21 percentage points** |
| **Micro Recall** | 87.29% (0.8729) | **99.45% (0.9945)** | **+12.16 percentage points** |
| **Micro Precision** | 95.76% (0.9576) | **96.77% (0.9677)** | **+1.01 percentage points** |
| **Multi-Label Exact Match** | 0.00% (0.0000) | **94.12% (0.9412)** | **+94.12 percentage points** |
| **False Negatives (FN)** | 23 | **1** | **-22 errors (95.65% reduction)** |
| **False Positives (FP)** | 7 | **6** | **-1 error** |

#### TECHNICAL INTERPRETATION
1. **Recall Surge (+12.16%):** Positive loss weighting (`pos_weight`) and extended 6-epoch training effectively eliminated false negative omissions.
2. **Multi-Label Breakthrough (+94.12%):** Extending sequence length to 128/256 tokens enabled BERT v2 to process full narrative context, resolving 16 out of 17 multi-label compound test cases correctly (compared to 0 in v1).

#### VIVA QUESTIONS
- **Q:** How much did BERT v2 improve over BERT v1?
- **Q:** Why did multi-label performance increase so dramatically?

#### VIVA ANSWERS
- **A:** Micro-F1 improved by **+6.76 percentage points** (91.33% to 98.09%), Exact Match improved by **+12.19 percentage points** (83.54% to 95.73%), and False Negatives dropped from 23 to 1.
- **A:** Expanding sequence length from 64 to 128/256 allowed the model to read entire multi-sentence narratives without truncating secondary incident mentions.

#### WHAT TO MEMORIZE
1. Micro-F1 gain: +6.76% (91.33% -> 98.09%).
2. Exact Match gain: +12.19% (83.54% -> 95.73%).
3. False Negative reduction: 23 -> 1 (95.65% reduction).

---

# PART 22 — BASELINE VS BERT

------------------------------------------------------------
### CONCEPT: Classical Baseline vs. Deep BERT Model Comparison
------------------------------------------------------------

#### WHAT IS IT?
A scientific comparison evaluating traditional linear TF-IDF models against fine-tuned deep learning BERT models on the same held-out test set ($N=164$).

#### COMPLETE MODEL COMPARISON TABLE

| Model Architecture | Exact Match | Micro Precision | Micro Recall | Micro F1-Score | Macro F1-Score | Multi-Label Exact Match | False Negatives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Linear SVM** | 95.12% | 97.83% | 96.72% | 97.27% | 96.69% | 82.35% | 6 | 4 |
| **TF-IDF + Logistic Regression** | 95.73% | **97.79%** | 97.79% | 97.79% | 97.35% | 88.24% | 4 | 4 |
| **BERT v1 (Baseline Fine-Tuned)** | 83.54% | 95.76% | 87.29% | 91.33% | 90.37% | 0.00% | 23 | 7 |
| **BERT v2 (Promoted Production)** | **95.73%** | 96.77% | **99.45%** | **98.09%** | **97.58%** | **94.12%** | **1** | **6** |

#### SCIENTIFIC ANALYSIS OF RESULTS
1. **Why Classical Models Performed Strongly:** The survivor dataset contains explicit, unambiguous lexical keyphrases (e.g. `"husband beat"`, `"dowry"`, `"fake account"`). TF-IDF n-gram features capture these surface keywords effectively.
2. **Where BERT v2 Superiority Lies:**
   - **Micro Recall (99.45% vs 97.79%):** BERT v2 achieved superior recall, minimizing missed harassment incidents (only 1 FN vs 4 in LogReg and 6 in SVM).
   - **Multi-Label Exact Match (94.12% vs 88.24%):** BERT v2 excels at capturing complex semantic interactions in multi-label compound incidents through deep bidirectional self-attention.

#### VIVA QUESTIONS
- **Q:** Is BERT always better than Logistic Regression on your dataset?
- **Q:** Why did Logistic Regression perform so well?

#### VIVA ANSWERS
- **A:** While Logistic Regression achieved strong Exact Match (95.73%), **BERT v2 outperformed all baselines in Micro-F1 (98.09%), Micro Recall (99.45%), and Multi-Label Exact Match (94.12%)**, minimizing critical false negatives.
- **A:** High surface-level keyword explicit in safety narratives allows TF-IDF n-grams to perform effectively for single-label cases.

#### WHAT TO MEMORIZE
1. Highest Micro-F1: BERT v2 (98.09%).
2. Highest Micro Recall: BERT v2 (99.45%, only 1 FN).
3. Highest Multi-Label Exact Match: BERT v2 (94.12%).

---

# PART 23 — NER

------------------------------------------------------------
### CONCEPT: Named Entity Recognition (NER) & BIO Schema
------------------------------------------------------------

#### WHAT IS IT?
Named Entity Recognition (NER) is a token classification task that identifies and extracts specific evidentiary spans (e.g. perpetrator relationship, physical location, duration/frequency, communication platform) from raw narrative text.

#### DIFFERENCE BETWEEN BERT CLASSIFICATION AND BERT NER
- **BERT Classification:** Operates at the **sequence level** using the `[CLS]` token to output 6 global incident category flags ($\mathbf{y} \in \{0, 1\}^6$).
- **BERT NER:** Operates at the **token level** (`AutoModelForTokenClassification`) to assign a specific entity tag to every subword token in the text.

#### WHERE IS IT IMPLEMENTED?
- Training Script: [`ml/ner/train_ner.py`](file:///d:/Abhera(Mini)/ml/ner/train_ner.py)
- Data Preparation: [`ml/ner/prepare_training_data.py`](file:///d:/Abhera(Mini)/ml/ner/prepare_training_data.py)
- Inference Script: [`ml/inference/predict_ner.py`](file:///d:/Abhera(Mini)/ml/inference/predict_ner.py)

#### ENTITY TYPES & BIO TAGGING SCHEMA
The project uses a 13-tag BIO (Begin, Inside, Outside) tagging schema:

| Entity Type | Meaning / Scope | BIO Tags | Example Span |
| :--- | :--- | :--- | :--- |
| **`PERP_REL`** | Perpetrator relationship to survivor. | `B-PERP_REL`, `I-PERP_REL` | `"husband"`, `"senior manager"` |
| **`LOCATION`** | Physical workplace/venue site. | `B-LOCATION`, `I-LOCATION` | `"office"`, `"home"`, `"metro station"` |
| **`TIME_FREQ`** | Time, duration, or frequency. | `B-TIME_FREQ`, `I-TIME_FREQ` | `"every evening"`, `"for 3 months"` |
| **`PLATFORM`** | Online app or channel. | `B-PLATFORM`, `I-PLATFORM` | `"Instagram"`, `"WhatsApp"` |
| **`O`** | Outside any named entity. | `O` | Non-entity words (`"and"`, `"was"`) |

*Special Loss Masking:* Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are assigned label index `-100` to mask them from loss calculations in PyTorch.

#### SUBWORD ALIGNMENT WITH OFFSET MAPPINGS
When tokenizing with WordPiece, a single word may split into multiple subwords (e.g. `"WhatsApp"` $\rightarrow$ `["whats", "##app"]`). `prepare_training_data.py` uses HuggingFace `return_offsets_mapping=True` to map character start/end offsets back to subword indices, assigning `B-<LABEL>` to the first subword and `I-<LABEL>` to subsequent subwords.

#### VIVA QUESTIONS
- **Q:** What is NER and why does ABHERA need it?
- **Q:** What is the BIO tagging schema?
- **Q:** How do you handle subwords during NER token classification?

#### VIVA ANSWERS
- **A:** NER extracts structured evidentiary entity spans (like perpetrator relationship or location) from narrative text. BERT classification tells us *what type of crime occurred*, while NER tells us *who, where, and how*.
- **A:** B = Beginning of entity, I = Inside entity, O = Outside entity.
- **A:** We use `return_offsets_mapping=True` to align subwords with character spans, assigning `B-` to the first subword, `I-` to remaining subwords, and `-100` to special tokens.

#### WHAT TO MEMORIZE
1. 4 core entity classes: `PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`.
2. Total BIO tags: 13 tags (4 B-tags, 4 I-tags, 1 O-tag, special tokens masked at -100).
3. Offset mapping tool: `return_offsets_mapping=True`.

---

# PART 24 — NER PERFORMANCE

------------------------------------------------------------
### CONCEPT: Named Entity Recognition Performance Evaluation
------------------------------------------------------------

#### WHAT IS IT?
The strict entity-level evaluation of the fine-tuned BERT-NER model conducted on 60 held-out test narrative records containing 66 gold entity spans.

#### WHERE IS IT STORED?
- Artifact File: `models/ner/test_metrics.json`
- Evaluation Report: [`ml/ner/training/ner_test_evaluation_report.txt`](file:///d:/Abhera(Mini)/ml/ner/training/ner_test_evaluation_report.txt)
- Consolidated Results: [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L231-L238)

#### DEFINITIVE NER EVALUATION METRICS

| Metric Name | Exact Mathematical Value | Scientific Percentage | Definition / Meaning |
| :--- | :---: | :---: | :--- |
| **Held-Out Test Records** | `60` | — | Number of test narratives evaluated. |
| **Gold Entity Spans** | `66` | — | Total annotated true entity spans. |
| **Entity-Level Micro Precision** | `1.0000` | **100.00%** | Fraction of predicted entity spans that are strictly correct. |
| **Entity-Level Micro Recall** | `0.9848` | **98.48%** | Fraction of gold entity spans correctly extracted (65 / 66 spans). |
| **Entity-Level Micro-F1** | `0.9924` | **99.24%** | Primary span-level harmonic mean performance metric. |
| **Active-Class Macro-F1** | `0.9900` | **99.00%** | Unweighted average F1 across active entity classes. |

#### NOMENCLATURE RULE
Do **NOT** call this metric "NER accuracy". In formal literature and viva defense, state explicitly:  
**"Entity-level Micro-F1 = 99.24%"**

#### VIVA QUESTIONS
- **Q:** What is your NER model performance?
- **Q:** How many gold entity spans were evaluated in the test set?

#### VIVA ANSWERS
- **A:** The fine-tuned BERT-NER model achieved an **entity-level Micro-F1 score of 99.24%** (Micro Precision = 100.00%, Micro Recall = 98.48%).
- **A:** 66 gold entity spans across 60 held-out test narrative records.

#### WHAT TO MEMORIZE
1. NER Entity-Level Micro-F1: 99.24%.
2. Precision: 100.00%; Recall: 98.48%.
3. Test set size: 60 records, 66 gold spans.

---

# PART 25 — QUESTION ENGINE

------------------------------------------------------------
### CONCEPT: Dynamic Questionnaire & Dialogue Orchestrator
------------------------------------------------------------

#### WHAT IS IT?
An interactive, context-aware dialogue engine that evaluates required entity dimensions against known information and selects non-repetitive follow-up questions to fill missing evidentiary gaps.

#### WHERE IS IT IMPLEMENTED?
- Main Orchestrator: [`backend/app/question_engine/engine.py`](file:///d:/Abhera(Mini)/backend/app/question_engine/engine.py)
- Rules & Category Requirements: [`backend/app/question_engine/rules.py`](file:///d:/Abhera(Mini)/backend/app/question_engine/rules.py)
- Question Selector: [`backend/app/question_engine/question_selector.py`](file:///d:/Abhera(Mini)/backend/app/question_engine/question_selector.py)
- Dynamic Generator: [`backend/app/question_engine/dynamic_generator.py`](file:///d:/Abhera(Mini)/backend/app/question_engine/dynamic_generator.py)

#### CATEGORY-TO-REQUIRED-ENTITY RULES

| Incident Category | Code | Required Information Dimensions |
| :--- | :---: | :--- |
| **Domestic Violence** | `DV` | `PERP_REL`, `INCIDENT_DETAILS`, `SAFETY_STATUS` |
| **Sexual Harassment** | `SH` | `PERP_REL`, `LOCATION`, `INCIDENT_DETAILS` |
| **Stalking** | `ST` | `PERP_REL`, `TIME_FREQ`, `PLATFORM`, `SAFETY_STATUS` |
| **Cyber Abuse** | `CA` | `PLATFORM`, `TIME_FREQ`, `INCIDENT_DETAILS` |
| **Workplace Harassment** | `WH` | `PERP_REL`, `LOCATION`, `INCIDENT_DETAILS` |
| **Other Violence** | `OV` | `INCIDENT_DETAILS`, `SAFETY_STATUS` |

#### CANONICAL PRIORITY ORDER
When multiple information dimensions are required, the engine resolves missing entities in strict priority order:  
`1. SAFETY_STATUS` $\rightarrow$ `2. PERP_REL` $\rightarrow$ `3. LOCATION` $\rightarrow$ `4. TIME_FREQ` $\rightarrow$ `5. PLATFORM` $\rightarrow$ `6. INCIDENT_DETAILS`.

#### QUESTION SELECTION LOGIC (`select_next_question`)
1. Filters out questions already asked (`asked_question_ids`).
2. Excludes inactive questions (`active == False`).
3. Must target an entity dimension currently in `missing_information`.
4. Must match active predicted categories or `General`.
5. Ranks remaining candidate questions by integer priority score; highest score wins.

#### NO-REPEAT & DYNAMIC CONTEXT MECHANISM
- **No-Repeat Guarantee:** Asked question IDs are tracked in `submission_question_answers` and excluded from selection.
- **Dynamic Personalization:** `generate_dynamic_question()` extracts phrase snippets from the user's initial narrative (e.g. *"your husband"*) to frame personalized follow-ups:  
  *Example Output:* *"Regarding the incident involving your husband, could you clarify where this took place?"*

#### VIVA QUESTIONS
- **Q:** How does the question engine decide what to ask next?
- **Q:** How do you prevent asking duplicate questions?

#### VIVA ANSWERS
- **A:** The engine subtracts known information (from NER and text heuristics) from the required information defined for active categories in `rules.py`, selecting the highest-priority question for the first missing dimension.
- **A:** The engine tracks asked question IDs in the `submission_question_answers` PostgreSQL table and filters them out during candidate selection.

#### WHAT TO MEMORIZE
1. Priority order: `SAFETY_STATUS` -> `PERP_REL` -> `LOCATION` -> `TIME_FREQ` -> `PLATFORM` -> `INCIDENT_DETAILS`.
2. Key files: `engine.py`, `rules.py`, `question_selector.py`, `dynamic_generator.py`.

---

# PART 26 — SESSION MANAGEMENT

------------------------------------------------------------
### CONCEPT: Session Lifecycle & Intake Conversation State
------------------------------------------------------------

#### WHAT IS IT?
Session management maintains multi-turn conversation state, connects successive user messages to a single incident intake submission, and tracks workflow progression.

#### WHERE IS IT IMPLEMENTED?
- ORM Model: [`backend/app/models/session.py`](file:///d:/Abhera(Mini)/backend/app/models/session.py)
- Service: [`backend/app/services/session_service.py`](file:///d:/Abhera(Mini)/backend/app/services/session_service.py)
- API Routes: [`backend/app/routes/session.py`](file:///d:/Abhera(Mini)/backend/app/routes/session.py)

#### CONVERSATION STATUS LIFECYCLE STATE MACHINE
The `conversation_status` column in the `sessions` table transitions through four explicit states:

```
[POST /api/session/start]
          │
          ▼
       ACTIVE  (Initial session creation; narrative entry)
          │
          ▼
    QUESTIONING  (Dynamic Question Engine asking follow-up questions)
          │
          ▼
    PROCESSING  (All required questions answered; mapping legal & support data)
          │
          ▼
     COMPLETED  (Report compiled, persisted to DB, and rendered)
```

#### IDENTIFIERS DEFINED
- **`session_id` (`SESS-xxxxxxxx`):** Public unique 64-character string identifying the active web chat session.
- **`submission_id` (`SUB-xxxxxxxx`):** Public unique 64-character string linking predictions, extracted entities, answers, and the final generated report to one legal incident case.

#### VIVA QUESTIONS
- **Q:** What are the four conversation statuses in ABHERA?
- **Q:** What is the difference between `session_id` and `submission_id`?

#### VIVA ANSWERS
- **A:** `ACTIVE`, `QUESTIONING`, `PROCESSING`, and `COMPLETED`.
- **A:** `session_id` identifies the web chat interaction session, while `submission_id` uniquely identifies the underlying legal incident submission and its generated report.

#### WHAT TO MEMORIZE
1. Lifecycle statuses: `ACTIVE` -> `QUESTIONING` -> `PROCESSING` -> `COMPLETED`.
2. Format: `SESS-xxxxxxxx` (session) and `SUB-xxxxxxxx` (submission).

---

# PART 27 — DATABASE

------------------------------------------------------------
### CONCEPT: PostgreSQL Database Schema & Relational ORM Architecture
------------------------------------------------------------

#### WHAT IS IT?
The persistent relational database system built on PostgreSQL and SQLAlchemy ORM, storing all application entity models, intake state, verified legal knowledge, support helplines, and generated reports.

#### WHERE IS IT IMPLEMENTED?
- Database Config: [`backend/app/database.py`](file:///d:/Abhera(Mini)/backend/app/database.py)
- ORM Models: [`backend/app/models/*.py`](file:///d:/Abhera(Mini)/backend/app/models)

#### VERIFIED DATABASE TABLES MATRIX (11 TABLES TOTAL)

| Table Name | ORM Model Class | Primary Key | Key Foreign Keys | Purpose & Stored Content | Written By | Read By |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `users` | `User` | `id` (Int) | None | Tracks anonymous user GUIDs (`user_guid`). | `SessionService` | `Session` relationship |
| `sessions` | `Session` | `id` (Int) | `user_id` -> `users.id` | Stores chat session state, messages, answers, predicted labels, entities JSON. | `SessionService`, `ChatService` | `ChatService`, Session API |
| `incident_submissions` | `IncidentSubmission` | `id` (Int) | `session_id` -> `sessions.session_id` | Master incident intake record containing narrative text, state, district. | `ChatService` | Report Generator, APIs |
| `conversation_messages` | `ConversationMessage` | `id` (Int) | `submission_id` -> `incident_submissions.submission_id` | Individual chat message history logs (`sender`, `message_text`). | `ChatService` | Chat API history |
| `predictions` | `Prediction` | `id` (Int) | `submission_id` -> `incident_submissions.submission_id` | Individual predicted category label scores (`label`, `confidence_score`). | `ChatService` | Evaluation & DB audit |
| `entities` | `Entity` | `id` (Int) | `submission_id` -> `incident_submissions.submission_id` | Extracted NER entity spans (`entity_type`, `entity_value`, `start_char`, `end_char`). | `ChatService` | Report Generator |
| `questions` | `Question` | `question_id` (String) | None | Master bank of seeded questionnaire questions (`category`, `required_entity`, `priority`). | `seed_questions.py` | Question Engine |
| `submission_question_answers` | `SubmissionQuestionAnswer` | `id` (Int) | `submission_id`, `question_id` | Tracks answered question state per incident submission (`question_id`, `answer`). | `engine.py` (`record_answer`) | Question Engine, Report Gen |
| `laws` | `Law` | `id` (Int) | None | Ground-truth statutory legal provisions (`act_name`, `section_number`, `section_text`, `applicable_label`). | `seed_data` / Importer | Legal Mapping, Anti-Hallucination |
| `support_services` | `SupportService` | `id` (Int) | None | Verified support service helplines (`name`, `contact_number`, `state`, `district`, `applicable_label`). | `seed_data` / Importer | Support Mapping, Anti-Hallucination |
| `reports` | `Report` | `id` (Int) | `submission_id` -> `incident_submissions.submission_id` | Final compiled 10-section incident report payload stored in `report_data` JSON. | `ReportGeneratorService` | Report API, PDF Generator |

#### VIVA QUESTIONS
- **Q:** How many database tables exist in ABHERA?
- **Q:** How do you store conversation messages and generated reports in PostgreSQL?

#### VIVA ANSWERS
- **A:** 11 database tables managed via SQLAlchemy ORM.
- **A:** `sessions.messages` and `reports.report_data` store structured JSON payloads, while individual message and report records maintain foreign key relationships to `incident_submissions`.

#### WHAT TO MEMORIZE
1. Total tables: 11 ORM entities.
2. Core relational anchor: `submission_id` in `incident_submissions`.

---

# PART 28 — LEGAL MAPPING

------------------------------------------------------------
### CONCEPT: Deterministic Legal Provision Retrieval
------------------------------------------------------------

#### WHAT IS IT?
A deterministic database lookup engine that maps predicted incident category codes (e.g. `DV`, `CA`) to verified statutory legal provisions stored in the PostgreSQL `laws` table.

#### WHERE IS IT IMPLEMENTED?
[`backend/app/services/legal_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/legal_mapping.py)

#### IMPORTANT CODE
```python
# Parameterized ORM Legal Query in backend/app/services/legal_mapping.py
raw_records = (
    db.query(Law)
    .filter(Law.applicable_label.in_(normalized_codes))
    .order_by(Law.id.asc())
    .all()
)
```

#### MATCHING LOGIC & MULTI-LABEL MERGING
1. Input category codes (e.g., `["DV", "CA"]`) are normalized via `LABEL_TO_APPLICABLE_CODE`.
2. Executes a parameterized SQL `IN` query filtering `Law.applicable_label.in_(normalized_codes)`.
3. If multiple categories are predicted, legal provisions for all active categories are merged and deduplicated using unique key `(act_name, section_number, applicable_label)`.
4. Mandatory fallback message if no match: `"Information not available in the provided knowledge base."`

#### VIVA QUESTIONS
- **Q:** How does ABHERA map predicted categories to laws?
- **Q:** Can the legal mapping module invent new laws?

#### VIVA ANSWERS
- **A:** It normalizes category codes and executes a parameterized ORM query on the `laws` table in PostgreSQL.
- **A:** **No.** It performs deterministic database retrieval. No legal text is generated by LLMs or AI models.

#### WHAT TO MEMORIZE
1. File path: `backend/app/services/legal_mapping.py`.
2. Fallback message: `"Information not available in the provided knowledge base."`

---

# PART 29 — SUPPORT MAPPING

------------------------------------------------------------
### CONCEPT: Location-Aware Support Service Retrieval
------------------------------------------------------------

#### WHAT IS IT?
A location-aware matching engine that retrieves verified support helplines from the `support_services` table based on predicted incident categories and survivor location (State and District).

#### WHERE IS IT IMPLEMENTED?
[`backend/app/services/support_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/support_mapping.py)

#### LOCATION RANKING ALGORITHM
Matches are ranked by location specificity score:
- **Exact District Match:** `location_score = 300`.
- **State Match:** `location_score = 200`.
- **National Helpline Fallback ("All India"):** `location_score = 100`.
- Non-matching local services are excluded (`score = 0`).

Mandatory fallback message if no match: `"No matching support service was found in the available support-services database."`

#### VIVA QUESTIONS
- **Q:** How does support service mapping handle location?
- **Q:** What is the fallback if no local district helpline is found?

#### VIVA ANSWERS
- **A:** It ranks support services using location scores: 300 for exact district, 200 for state, and 100 for national helplines.
- **A:** It falls back to national 24x7 helplines (e.g. NCW 7827170170 or ERSS 112).

#### WHAT TO MEMORIZE
1. Location scores: District (300) > State (200) > National (100).
2. Fallback message: `"No matching support service was found in the available support-services database."`

---

# PART 30 — ANTI-HALLUCINATION

------------------------------------------------------------
### CONCEPT: Zero-Hallucination Safety & Provenance Safeguard
------------------------------------------------------------

#### WHAT IS IT?
A strict validation and gating service that verifies all legal provisions and support services against empirical PostgreSQL database records before allowing them into final reports.

#### WHERE IS IT IMPLEMENTED?
[`backend/app/services/anti_hallucination.py`](file:///d:/Abhera(Mini)/backend/app/services/anti_hallucination.py)

#### CORE ARCHITECTURAL PRINCIPLE
> **"The model predicts and extracts; the database verifies and provides the final legal and support information."**

#### VALIDATION CHECKS EXECUTED (`validate_and_gate_results`)
1. **Legal Validation (`validate_legal_record`):** Queries `laws` table for exact matching `act_name` and `section_number`. Verifies `section_text` matches database verbatim. Any modified or un-found record is rejected with: `"Fabricated legal record rejected"`.
2. **Support Validation (`validate_support_record`):** Queries `support_services` table for exact matching `name` and `contact_number`. Any un-found helpline is rejected with: `"Fabricated support service rejected"`.

#### VIVA QUESTIONS
- **Q:** How do you prevent AI hallucinations in legal advice?
- **Q:** What happens if a legal provision does not exist in your database?

#### VIVA ANSWERS
- **A:** AI models only classify incident categories and extract entities. All legal provisions and support services are fetched directly from PostgreSQL and validated verbatim by `AntiHallucinationService`.
- **A:** The system rejects the entry and displays the verified fallback message: `"Information not available in the provided knowledge base."`

#### WHAT TO MEMORIZE
1. Core principle: Models predict & extract; Database verifies & provides.
2. File path: `backend/app/services/anti_hallucination.py`.

---

# PART 31 — REPORT GENERATION

------------------------------------------------------------
### CONCEPT: Executive Incident Report Compilation
------------------------------------------------------------

#### WHAT IS IT?
The service that compiles all multi-module outputs into a structured 10-section JSON report schema and persists it to the `reports` table.

#### WHERE IS IT IMPLEMENTED?
[`backend/app/services/report_generator.py`](file:///d:/Abhera(Mini)/backend/app/services/report_generator.py)

#### THE 10 CONCEPTUAL REPORT SECTIONS
1. **Incident Summary:** Survivor's original incident narrative text string.
2. **Identified Incident Type:** BERT predicted category labels (`labels`).
3. **Extracted Information:** NER extracted entities (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`).
4. **User-Provided Details:** Questionnaire answers from `submission_question_answers`.
5. **Relevant Legal Information:** Database-verified statutory acts and sections.
6. **Suggested Next Steps:** Standard practical safety and evidence preservation steps.
7. **Available Support Services:** Location-ranked verified support helplines.
8. **Evidence / Information Notes:** Extracted evidence notes and user mentions.
9. **Model Prediction Information:** Transparency metadata (`bert-base-uncased`, data provenance).
10. **Disclaimer:** Legal disclaimer stating the report is for informational assistance.

#### VIVA QUESTIONS
- **Q:** What are the 10 sections of the ABHERA incident report?
- **Q:** Is the report text generated by GPT or an LLM?

#### VIVA ANSWERS
- **A:** Incident Summary, Identified Incident Type, Extracted Info, User Details, Legal Info, Next Steps, Support Services, Evidence Notes, Model Info, and Disclaimer.
- **A:** **No.** The report is non-generative. It assembles model predictions, NER extractions, and verified database records into a structured template.

#### WHAT TO MEMORIZE
1. Total report sections: 10 conceptual sections.
2. Storage table: PostgreSQL `reports.report_data` JSON column.

---

# PART 32 — REPORT API

------------------------------------------------------------
### CONCEPT: RESTful API Endpoints & Request/Response Contracts
------------------------------------------------------------

#### WHAT IS IT?
The set of FastAPI REST endpoints exposing session initialization, chat message processing, and report generation/retrieval.

#### WHERE IS IT IMPLEMENTED?
- [`backend/app/routes/session.py`](file:///d:/Abhera(Mini)/backend/app/routes/session.py)
- [`backend/app/routes/chat.py`](file:///d:/Abhera(Mini)/backend/app/routes/chat.py)
- [`backend/app/routes/report.py`](file:///d:/Abhera(Mini)/backend/app/routes/report.py)

#### ENDPOINT CONTRACTS TABLE

| Method | Path | Input Validation | Primary Processing | Output Response | Status Codes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/session/start` | Optional `user_id`, `initial_narrative` | `SessionService.create_session()` | `SessionResponse` (`session_id`, `submission_id`) | `200 OK`, `500 Error` |
| `GET` | `/api/session/{session_id}` | Path param `session_id` string | `SessionService.get_session()` | `SessionResponse` current state | `200 OK`, `400 Bad`, `404 Not Found` |
| `POST` | `/api/chat/message` | `session_id`, `message` string | `ChatService.process_message()` (BERT + NER + Engine) | `ChatMessageResponse` (bot reply, labels, entities) | `200 OK`, `400 Bad`, `404 Not Found` |
| `POST` | `/api/report/generate` | `submission_id` string | `generate_incident_report()` | `ReportResponse` 10-section JSON | `200 OK`, `400 Bad`, `404 Not Found` |
| `GET` | `/api/report/{submission_id}`| Path param `submission_id` | Queries `Report` table in PostgreSQL | `ReportResponse` JSON payload | `200 OK`, `400 Bad`, `404 Not Found` |

---

# PART 33 — FRONTEND

------------------------------------------------------------
### CONCEPT: React Web Application & Client-Side Architecture
------------------------------------------------------------

#### WHAT IS IT?
The single-page web frontend built with React 18 and Vite, managing user interaction, multi-turn chat intake, and report rendering.

#### WHERE IS IT IMPLEMENTED?
- Repository Directory: `frontend/`
- Central State Context: [`frontend/src/context/ConversationContext.jsx`](file:///d:/Abhera(Mini)/frontend/src/context/ConversationContext.jsx)
- API Client: [`frontend/src/services/api.js`](file:///d:/Abhera(Mini)/frontend/src/services/api.js)
- Key Pages: `Home.jsx`, `Chatbot.jsx`, `Processing.jsx`, `Report.jsx`

#### SESSION STORAGE PERSISTENCE
`ConversationContext.jsx` synchronizes active session keys (`abhera_session_id` and `abhera_submission_id`) with browser `sessionStorage`. This preserves active survivor intake context across page refreshes while ensuring automatic session clearance when the browser tab is closed.

---

# PART 34 — PDF GENERATION

------------------------------------------------------------
### CONCEPT: Client-Side PDF Report Generation & Styling
------------------------------------------------------------

#### WHAT IS IT?
The client-side PDF rendering engine built using `jsPDF` inside `frontend/src/pages/Report.jsx`.

#### WHERE IS IT IMPLEMENTED?
[`frontend/src/pages/Report.jsx`](file:///d:/Abhera(Mini)/frontend/src/pages/Report.jsx#L361-L800) (`handleDownload` function)

#### DESIGN TOKENS (`ABHERA_COLORS`)
PDF generation uses exact master design tokens for visual parity with the web interface:
- **`page` (`#F7F5F0`):** Warm Ivory document background fill.
- **`navy` (`#101820`):** Navy Black header and footer structural banners.
- **`primary` (`#1E4F8A`):** Primary Legal Navy Blue accents and checkmarks.
- **`section` (`#F4F8FC`):** Pale Blue section container background fill.
- **`gold` (`#B89A5A`):** Premium Gold separator accent lines.

#### VECTOR CHECKMARK DRAWER
`drawCheckmark(doc, cx, cy)` draws checkmark badges using vector line primitive calls (`setLineWidth(0.65)`, `line()`), guaranteeing zero broken font characters or encoding errors across OS platforms.

---

# PART 35 — TESTING

------------------------------------------------------------
### CONCEPT: Test Suite Coverage & System Verification
------------------------------------------------------------

#### WHAT IS IT?
The unit and integration testing suite verifying backend services, question engine logic, legal/support mapping, anti-hallucination gating, and API endpoints.

#### WHERE IS IT IMPLEMENTED?
- Backend Test Suite Directory: `backend/tests/`
- Test Files: `test_chat_api.py`, `test_dynamic_question_generation.py`, `test_fpm_incident.py`, `test_hallucination.py`, `test_legal_mapping.py`, `test_multilabel_classification.py`, `test_question_engine.py`, `test_report_api.py`, `test_report_generator.py`, `test_session_api.py`, `test_support_mapping.py`.

#### SUMMARY OF REGRESSION TEST RESULTS
- **Total Tests Executed:** `154`.
- **Passed Tests:** `154`.
- **Failed Tests:** `0`.
- **Pass Rate:** **100.0%**.
- **Execution Command:** `python -m unittest discover tests` (from `backend/`).

---

# PART 36 — LEAKAGE AUDIT

------------------------------------------------------------
### CONCEPT: Dataset Cross-Split Leakage & Similarity Audit
------------------------------------------------------------

#### WHAT IS IT?
A forensic audit checking for data contamination across training (760), validation (162), and test (164) dataset splits.

#### WHERE IS IT IMPLEMENTED?
[`ml/evaluation/leakage_audit/run_leakage_audit.py`](file:///d:/Abhera(Mini)/ml/evaluation/leakage_audit/run_leakage_audit.py)

#### AUDIT METRICS & FINDINGS
1. **Cross-Split Hash Duplicates:** `0` exact MD5 narrative hash matches across splits.
2. **Cross-Split Text Overlaps:** `0` exact string overlaps between train and test sets.
3. **Keyword Leakage:** Verified statutory category labels (`"DV"`, `"SH"`, etc.) do not appear verbatim inside raw narrative texts.

---

# PART 37 — END-TO-END VALIDATION

------------------------------------------------------------
### CONCEPT: 11-Subsystem End-to-End System Validation
------------------------------------------------------------

#### WHAT IS IT?
A complete integration audit testing all 11 core functional modules end-to-end.

#### WHERE IS IT STORED?
- Report Artifact: [`ml/evaluation/abhera_end_to_end_validation.md`](file:///d:/Abhera(Mini)/ml/evaluation/abhera_end_to_end_validation.md)
- JSON Benchmark: [`ml/evaluation/abhera_end_to_end_validation.json`](file:///d:/Abhera(Mini)/ml/evaluation/abhera_end_to_end_validation.json)

#### SUBSYSTEM VALIDATION MATRIX

| Subsystem # | Subsystem Name | Validation Status | Verified Output / Behavior |
| :--- | :--- | :---: | :--- |
| 1 | BERT Backend Inference | **PASS** | Correct 6D probability predictions and thresholding. |
| 2 | BERT-NER Coexistence | **PASS** | Concurrent entity extraction without tensor shape conflicts. |
| 3 | Dynamic Question Engine | **PASS** | Correct missing info resolution & priority question selection. |
| 4 | Legal Knowledge Base | **PASS** | Exact retrieval of IPC/BNS sections from PostgreSQL `laws`. |
| 5 | Support Service Mapper | **PASS** | Correct location-based helpline ranking (district/state/national). |
| 6 | Executive Report Assembly | **PASS** | Complete 10-section JSON schema compilation. |
| 7 | PDF Renderer Engine | **PASS** | Successful 7-page PDF rendering via `jsPDF`. |
| 8 | Database Consistency | **PASS** | Relational integrity maintained across all 11 ORM tables. |
| 9 | FastAPI Route Integration | **PASS** | Clean response serialization across `/api/chat` and `/api/report`. |
| 10 | Anti-Hallucination Guard | **PASS** | Rejection of un-found legal/support records; fallback messages. |
| 11 | Backend Regression Suite | **PASS** | 154 / 154 unit tests passing (100% pass rate). |

**FINAL END-TO-END STATUS: PASS**

---

# PART 38 — FINAL MODEL PROMOTION

------------------------------------------------------------
### CONCEPT: Experimental Model Promotion & Production Deployment
------------------------------------------------------------

#### WHAT IS IT?
The formal audit and promotion workflow that evaluated experimental BERT v2 against baseline BERT v1 and promoted v2 into the production models directory.

#### WHERE IS IT DOCUMENTED?
[`ml/evaluation/bert_v2_production_promotion.md`](file:///d:/Abhera(Mini)/ml/evaluation/bert_v2_production_promotion.md)

#### PROMOTION WORKFLOW
1. Experimental model `models/bert_multilabel_experiment_v2/` evaluated on held-out test set ($N=164$).
2. Demonstrated +6.76% Micro-F1 gain and 95.65% reduction in false negatives over baseline `models/bert_multilabel/`.
3. Validated zero test-set leakage in threshold selection.
4. Promoted `models/bert_multilabel_experiment_v2/` as the primary production model loaded by `predict_bert.py`.

---

# PART 39 — FORMULAS

### Important Implementation Mathematical Formulations

#### 1. Multi-Label Target Vector
$$\mathbf{y} = [y_{\text{DV}}, y_{\text{SH}}, y_{\text{ST}}, y_{\text{CA}}, y_{\text{WH}}, y_{\text{OV}}]^T \in \{0, 1\}^6$$

#### 2. Independent Sigmoid Activation
$$p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \quad \text{for } k \in \{1 \dots 6\}$$

#### 3. Label-Specific Threshold Decision Rule
$$\hat{y}_k = \begin{cases} 1 & \text{if } p_k \ge \tau_k \\ 0 & \text{otherwise} \end{cases}$$
Where optimal validation thresholds are:  
$$\tau = [\tau_{\text{DV}}=0.73, \tau_{\text{SH}}=0.50, \tau_{\text{ST}}=0.79, \tau_{\text{CA}}=0.45, \tau_{\text{WH}}=0.29, \tau_{\text{OV}}=0.63]$$

#### 4. Positive-Class Weighted Binary Cross-Entropy Loss
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{B \cdot K} \sum_{i=1}^{B} \sum_{k=1}^{K} \left[ w_k \cdot y_{i,k} \cdot \ln(\sigma(z_{i,k})) + (1 - y_{i,k}) \cdot \ln(1 - \sigma(z_{i,k})) \right]$$
Where positive weight $w_k = \frac{N_{\text{total}} - N_{\text{pos}, k}}{N_{\text{pos}, k}}$.

#### 5. Sublinear TF-IDF Formulation
$$\text{TF}(t, d) = 1 + \ln(\text{tf}(t, d)) \quad \text{for } \text{tf}(t, d) > 0$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### 6. Micro F1-Score Formulation
$$\text{Precision}_{\text{micro}} = \frac{\sum_{k=1}^K \text{TP}_k}{\sum_{k=1}^K \text{TP}_k + \sum_{k=1}^K \text{FP}_k}, \quad \text{Recall}_{\text{micro}} = \frac{\sum_{k=1}^K \text{TP}_k}{\sum_{k=1}^K \text{TP}_k + \sum_{k=1}^K \text{FN}_k}$$
$$\text{Micro-F1} = \frac{2 \times \text{Precision}_{\text{micro}} \times \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}$$

---

# PART 40 — ALGORITHMS / PSEUDOCODE

*(Implementation-faithful pseudocode matching repository source code)*

### Algorithm 1 — Reproducible Dataset Preparation & Splitting
```
Input : Raw CSV file path, Random Seed S = 42
Output: Split CSV files (train_data.csv, val_data.csv, test_data.csv)

1. Load df = ReadCSV(file_path)
2. Filter valid_mask = df["narrative_text"].notnull() and df["narrative_text"].strip() != ""
3. clean_df = df[valid_mask]
4. Construct clean_df["label_vector"] = clean_df[["DV", "SH", "ST", "CA", "WH", "OV"]]
5. Set random seed(S)
6. shuffled_df = clean_df.sample(frac=1.0, random_state=S)
7. Slice train_df = shuffled_df[0 : 760]       (70%)
8. Slice val_df   = shuffled_df[760 : 922]      (15%)
9. Slice test_df  = shuffled_df[922 : 1086]     (15%)
10. Save train_df, val_df, test_df to disk
```

### Algorithm 2 — BERT Multi-Label Forward Pass & Threshold Inference
```
Input : Raw narrative text T, Pretrained BERT Encoder, Linear Head W, Thresholds τ
Output: Predicted category codes list

1. tokens = Tokenizer.encode(T, truncation=True, max_length=256, padding="max_length")
2. input_ids = tokens.input_ids
3. attention_mask = tokens.attention_mask
4. hidden_states = BERT_Encoder(input_ids, attention_mask)
5. h_cls = hidden_states[:, 0, :]   // [CLS] token representation (768-dim)
6. logits z = Linear_Head(Dropout(h_cls))  // [6-dim logits]
7. probabilities p = Sigmoid(z)            // [6-dim probabilities]
8. predicted_labels = []
9. For k = 0 to 5:
10.    if p[k] >= τ[k]:
11.        predicted_labels.append(LABEL_NAMES[k])
12. Return predicted_labels
```

### Algorithm 3 — Threshold Optimization Grid Search
```
Input : Validation probabilities P_val (N_val x 6), Ground truth Y_val (N_val x 6)
Output: Optimal thresholds τ_opt (6-dim vector)

1. For k = 0 to 5:
2.     best_f1 = 0.0, best_tau = 0.50
3.     For tau in range(0.10, 0.90, step=0.01):
4.         y_pred = (P_val[:, k] >= tau).astype(int)
5.         f1 = ComputeF1(Y_val[:, k], y_pred)
6.         if f1 > best_f1:
7.             best_f1 = f1
8.             best_tau = tau
9.     τ_opt[k] = best_tau
10. Return τ_opt
```

---

# PART 41 — COMPLETE WORKFLOW EXAMPLE

### End-to-End Execution Trace

1. **User Input:** Survivor types into React UI:  
   *"My husband hit me at home and threatens to post my private photos online."*
2. **Session Creation:** React invokes `POST /api/session/start`. Server returns `session_id = "SESS-a1b2c3d4"`, `submission_id = "SUB-e5f6g7h8"`.
3. **Chat Message Submission:** React sends `POST /api/chat/message`.
4. **BERT Classification:** `predict_bert.py` outputs probabilities:  
   `DV`: $0.8778 \ge 0.73 \rightarrow \mathbf{1}$, `CA`: $0.8443 \ge 0.45 \rightarrow \mathbf{1}$, others $< \tau$.  
   *Predicted Labels:* `["Domestic Violence", "Cyber Abuse"]`.
5. **NER Entity Extraction:** `predict_ner.py` extracts:  
   `PERP_REL`: `"husband"`, `LOCATION`: `"home"`.
6. **Question Engine Execution:** Required info for `DV` + `CA` = `[SAFETY_STATUS, PERP_REL, LOCATION, TIME_FREQ, PLATFORM, INCIDENT_DETAILS]`. Known = `[PERP_REL, LOCATION, INCIDENT_DETAILS]`. Missing = `[SAFETY_STATUS, TIME_FREQ, PLATFORM]`.  
   *Selected Follow-up Question:* `SAFETY_STATUS` (Priority #1):  
   *"You mentioned the incident involving your husband. Are you currently in immediate danger?"*
7. **User Reply:** Survivor answers: *"I am safe right now with a friend."*
8. **Report Generation:** `generate_incident_report()` compiles 10 sections:
   - *Legal Query:* Looks up `DV` (IPC 498A/BNS 85) + `CA` (IT Act 66E, 67).
   - *Support Query:* Looks up helplines for `DV` and `CA`.
   - *Anti-Hallucination:* Verifies exact match in PostgreSQL `laws` and `support_services`.
9. **PDF Export:** User clicks "Download Official Report PDF". `jsPDF` renders 7-page PDF with warm ivory background (`#F7F5F0`) and legal navy headers (`#101820`).

---

# PART 42 — ERROR HANDLING

### System Fallback & Exception Matrix

| Module / Layer | Potential Error Condition | Fallback Mechanism / Response |
| :--- | :--- | :--- |
| **Input API** | Empty message or missing `session_id` | Returns HTTP `400 Bad Request` (`"Session ID cannot be empty"`). |
| **Session API** | Non-existent `session_id` | Returns HTTP `404 Not Found` (`"Session with ID not found"`). |
| **ML Predictor** | Torch/Model loading failure | Gracefully logs warning; returns empty predictions without crashing API. |
| **Legal Mapping** | No laws match predicted categories | Returns fallback: `"Information not available in the provided knowledge base."` |
| **Support Mapping** | No helplines match location/category | Returns fallback: `"No matching support service was found..."` |
| **Anti-Hallucination** | Fabricated/modified legal string detected | Gates out invalid record; logs rejection reason; returns verified fallback. |
| **Report API** | Report generation exception | Catches error; rolls back DB transaction; returns HTTP `500`. |

---

# PART 43 — SECURITY / PRIVACY

### Implemented Security & Privacy Safeguards
- **CORS Middleware:** Configured in `backend/app/main.py` restricting allowed HTTP origins (`settings.CORS_ORIGINS`).
- **SQL Injection Prevention:** All database queries execute via SQLAlchemy ORM parameterized queries (`filter(Law.applicable_label.in_(...))`), preventing SQL injection.
- **Privacy-Preserving Intake:** Users are assigned anonymous GUIDs (`is_anonymous=True`). No mandatory personally identifiable information (PII) like national ID numbers is required.
- **Session Scoping:** Session IDs are managed in `sessionStorage`, auto-clearing when browser tab closes.

---

# PART 44 — LIMITATIONS

### Empirical System Limitations
1. **Dataset Size:** Trained on 1,086 narrative records. While achieving high accuracy on test splits, expanding to 10,000+ real-world narratives will improve edge-case generalization.
2. **Fixed Category Boundary:** Restricted to 6 defined statutory harassment classes (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
3. **Language Scope:** Current fine-tuned model operates on English narratives (`bert-base-uncased`). Multilingual support (Hindi, regional languages) requires `bert-base-multilingual-cased` fine-tuning.
4. **Local Support Coverage:** `support_services` table contains 12 verified helplines. Expanding support coverage requires continuous database seeding.

---

# PART 45 — FUTURE ENHANCEMENTS

### Scope of Future Extensions
- **Multilingual Transformer Models:** Fine-tuning `mBERT` or `XLM-RoBERTa` for multi-lingual survivor intake in Hindi, Tamil, Telugu, and regional languages.
- **Speech-to-Text Voice Intake:** Integrating Whisper API for voice-assisted survivor narrative reporting.
- **Expanded Legal Knowledge Base:** Ingesting state-specific legal amendments and municipal protection officer contact directories.

---

# PART 46 — PAPER-READY METHODOLOGY

```markdown
### 3. METHODOLOGY

#### 3.1 Problem Formulation and Architecture
Let $\mathcal{D} = \{(x_i, \mathbf{y}_i)\}_{i=1}^N$ denote a corpus of $N=1,086$ survivor-submitted incident narratives, where $x_i$ represents an unstructured text narrative and $\mathbf{y}_i = [y_{i,1}, \dots, y_{i,K}]^T \in \{0, 1\}^K$ represents a multi-hot binary label vector across $K=6$ statutory incident categories: Domestic Violence (DV), Sexual Harassment (SH), Stalking (ST), Cyber Abuse (CA), Workplace Harassment (WH), and Other Violence (OV). The objective is to learn a mapping function $f: x_i \to \hat{\mathbf{y}}_i$ while concurrently extracting token-level evidentiary entity spans $\mathcal{E}_i$ and retrieving verified statutory provisions from a relational database without generating hallucinated legal content.

#### 3.2 Dataset Preparation and Partitioning
The dataset was validated to ensure schema compliance and text integrity. Null entries and whitespace narratives were eliminated. The validated corpus ($N=1,086$) was partitioned into training (70%, 760 samples), validation (15%, 162 samples), and held-out test (15%, 164 samples) splits using a fixed pseudorandom seed ($S=42$). Preprocessing preserved full syntactic structure; stop-word removal, stemming, and lemmatization were omitted to maintain sentence syntax for transformer self-attention mechanisms.

#### 3.3 Deep Multi-Label BERT Classification
Sequence classification was implemented by fine-tuning `bert-base-uncased` (110M parameters). Input narratives were tokenized with subword WordPiece tokenization, appending classification `[CLS]` and separator `[SEP]` tokens with a maximum sequence length of 128 (training) and 256 (inference). The final hidden vector $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$ was fed through a dropout layer ($p=0.10$) into a linear classification head producing 6 raw logits $\mathbf{z} \in \mathbb{R}^6$. To address class imbalance, models were trained using Binary Cross-Entropy with Logits loss incorporating positive class weights $w_k = (N - N_k)/N_k$. Optimization was executed using AdamW ($lr=3\times 10^{-5}$, weight decay $0.01$, batch size $32$) over 6 epochs with a 10% linear learning rate warmup.

#### 3.4 Decision Threshold Optimization
Rather than assuming a default decision threshold of $\tau=0.50$, category-specific thresholds $\tau_k \in [0.10, 0.90]$ were optimized via grid search strictly on validation probabilities ($N=162$) to maximize per-label F1-scores. The resulting optimal thresholds ($\tau_{\text{DV}}=0.73, \tau_{\text{SH}}=0.50, \tau_{\text{ST}}=0.79, \tau_{\text{CA}}=0.45, \tau_{\text{WH}}=0.29, \tau_{\text{OV}}=0.63$) were frozen prior to evaluation on the held-out test set ($N=164$).

#### 3.5 Token-Level Named Entity Recognition
Entity extraction was formulated as a token classification task using `AutoModelForTokenClassification` fine-tuned over a 13-tag BIO schema (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`). HuggingFace offset mappings (`return_offsets_mapping=True`) were used to align WordPiece subword tokens back to original character spans, assigning a loss masking index of -100 to special tokens.

#### 3.6 Deterministic Database Grounding & Anti-Hallucination
To guarantee zero legal hallucination, ABHERA separates prediction from retrieval. Predicted incident categories trigger parameterized SQL queries against PostgreSQL `laws` and `support_services` tables. An Anti-Hallucination Service validates returned records against database entries verbatim before serializing the final 10-section report.
```

---

# PART 47 — PAPER-READY RESULTS

```markdown
### 4. EXPERIMENTAL RESULTS AND EVALUATION

#### 4.1 Comparative Model Performance
Table 1 presents the performance comparison between classical TF-IDF baselines, initial BERT v1, and promoted BERT v2 on the held-out test set ($N=164$).

**Table 1: Comparative Model Performance on Held-Out Test Set (N=164)**
| Model Architecture | Exact Match | Micro Precision | Micro Recall | Micro F1-Score | Macro F1-Score | Multi-Label Exact Match | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + Linear SVM | 95.12% | 97.83% | 96.72% | 97.27% | 96.69% | 82.35% | 6 |
| TF-IDF + Logistic Regression | 95.73% | 97.79% | 97.79% | 97.79% | 97.35% | 88.24% | 4 |
| BERT v1 (Baseline Fine-Tuned) | 83.54% | 95.76% | 87.29% | 91.33% | 90.37% | 0.00% | 23 |
| **BERT v2 (Promoted Production)** | **95.73%** | **96.77%** | **99.45%** | **98.09%** | **97.58%** | **94.12%** | **1** |

Promoted BERT v2 achieved superior overall performance, securing a **Micro F1-Score of 98.09%**, **Micro Recall of 99.45%**, and **Multi-Label Exact Match of 94.12%**, reducing false negative errors from 23 down to a single incident.

#### 4.2 Per-Label BERT v2 Evaluation
Table 2 details the category-specific performance breakdown and optimal decision thresholds.

**Table 2: Per-Label Performance Breakdown for Promoted BERT v2**
| Category Code | Optimal Threshold ($\tau_k$) | Support | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Domestic Violence (DV) | 0.73 | 35 | 35 | 2 | 0 | 127 | 94.59% | 100.00% | 97.22% |
| Sexual Harassment (SH) | 0.50 | 45 | 44 | 0 | 1 | 119 | 100.00% | 97.78% | 98.88% |
| Stalking (ST) | 0.79 | 32 | 32 | 0 | 0 | 132 | 100.00% | 100.00% | 100.00% |
| Cyber Abuse (CA) | 0.45 | 39 | 39 | 1 | 0 | 124 | 97.50% | 100.00% | 98.73% |
| Workplace Harassment (WH) | 0.29 | 19 | 19 | 2 | 0 | 143 | 90.48% | 100.00% | 95.00% |
| Other Violence (OV) | 0.63 | 11 | 11 | 1 | 0 | 152 | 91.67% | 100.00% | 95.65% |
| **OVERALL (MICRO)** | — | **181** | **180** | **6** | **1** | **797** | **96.77%** | **99.45%** | **98.09%** |

#### 4.3 Named Entity Recognition Evaluation
Evaluated on 60 held-out test records containing 66 gold entity spans, the BERT-NER token classifier achieved an **entity-level Micro Precision of 100.00%**, **Micro Recall of 98.48%**, and **entity-level Micro-F1 of 99.24%**.
```

---

# PART 48 — VIVA PREPARATION

*(100 Comprehensive Viva Questions & Answers Organized into Categories)*

### Category A: Project Basics & Motivation
1. **Q:** What is the full form of ABHERA?  
   **A:** Adaptive BERT-based Harassment Entity Recognition & Assistance.
2. **Q:** What is the primary objective of ABHERA?  
   **A:** To provide survivors of harassment with automated, zero-hallucination legal guidance and verified support helpline mapping.
3. **Q:** Who are the target users?  
   **A:** Survivors of harassment, legal aid volunteers, NGOs, legal counselors, and institutional POSH IC committees.
4. **Q:** What problem does ABHERA solve that commercial LLMs cannot?  
   **A:** Commercial LLMs hallucinate legal provisions and fail on privacy. ABHERA uses fine-tuned BERT for classification and deterministic PostgreSQL retrieval for zero legal hallucination.
5. **Q:** Does ABHERA replace lawyers?  
   **A:** No. ABHERA provides informational assistance and structured report generation, explicitly disclaiming binding legal representation.

### Category B: Dataset & Splitting
6. **Q:** What is the total size of the ABHERA dataset?  
   **A:** 1,086 narrative records.
7. **Q:** What split ratio was used?  
   **A:** 70% Training (760 rows), 15% Validation (162 rows), and 15% Test (164 rows).
8. **Q:** What random seed was used for splitting?  
   **A:** Random seed `42`.
9. **Q:** Why was standard single-label stratification omitted?  
   **A:** Multi-label target vectors $\mathbf{y} \in \{0, 1\}^6$ contain overlapping label combinations that standard single-label stratification cannot split.
10. **Q:** Were there missing narrative values in the raw CSV?  
    **A:** Zero. All 1,086 records contained valid text.

### Category C: Preprocessing & Data Validation
11. **Q:** Where is data validation implemented?  
    **A:** In `ClassificationDataPreparer.prepare()` in `ml/preprocessing/prepare_classification.py`.
12. **Q:** Did you remove stop words or perform stemming?  
    **A:** No. Preserving stop words and punctuation maintains full sentence syntax required by Transformer self-attention.
13. **Q:** How is lowercasing handled?  
    **A:** Lowercasing is handled automatically by the `bert-base-uncased` WordPiece tokenizer.
14. **Q:** How are special characters handled?  
    **A:** Special characters are retained in raw text and processed by the subword tokenizer.
15. **Q:** How are empty narrative entries handled?  
    **A:** Checked via `notnull()` and `.str.strip() != ""` prior to splitting.

### Category D: Feature Extraction & Feature Selection
16. **Q:** How are text features extracted for classical baselines?  
    **A:** Using `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=5000)`.
17. **Q:** Did you use explicit feature selection algorithms like SelectKBest or Chi2?  
    **A:** **No.** No supervised feature selection algorithms are implemented in the project.
18. **Q:** Is `max_features=5000` considered supervised feature selection?  
    **A:** No. It is an unsupervised vocabulary bound based purely on corpus term frequency.
19. **Q:** How does BERT extract features?  
    **A:** BERT uses 12 Transformer self-attention layers to dynamically compute 768-dimensional contextual embeddings.
20. **Q:** Did you use PCA or SVD for dimensionality reduction?  
    **A:** No.

### Category E: Classical Baselines & TF-IDF
21. **Q:** What baseline models did you train?  
    **A:** TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
22. **Q:** What was the Micro-F1 of Logistic Regression?  
    **A:** 97.79%.
23. **Q:** What was the Micro-F1 of Linear SVM?  
    **A:** 97.27%.
24. **Q:** Why did classical baselines perform so well?  
    **A:** Harassment narratives contain explicit lexical keywords (e.g. `"husband"`, `"slapped"`), which TF-IDF n-grams capture effectively.
25. **Q:** How did baselines handle multi-label outputs?  
    **A:** Using `OneVsRestClassifier`, fitting 6 independent binary linear classifiers.

### Category F: BERT Concepts & Pretraining
26. **Q:** What does BERT stand for?  
    **A:** Bidirectional Encoder Representations from Transformers.
27. **Q:** Which pretrained base model was used?  
    **A:** `bert-base-uncased`.
28. **Q:** How many parameters does `bert-base-uncased` have?  
    **A:** 110 Million parameters.
29. **Q:** What is the hidden layer dimension of BERT?  
    **A:** 768 dimensions.
30. **Q:** How many Transformer encoder layers and attention heads exist in BERT base?  
    **A:** 12 layers and 12 attention heads.

### Category G: BERT Input Processing & Tokenization
31. **Q:** What tokenizer does BERT use?  
    **A:** WordPiece tokenization with a vocabulary of 30,522 tokens.
32. **Q:** What is the `[CLS]` token and its index?  
    **A:** Classification token at index 0 (Token ID 101).
33. **Q:** What is the `[SEP]` token?  
    **A:** Separator token at the end of the sequence (Token ID 102).
34. **Q:** What sequence length was used during BERT v2 training?  
    **A:** `max_length = 128` tokens.
35. **Q:** What sequence length was used for production inference?  
    **A:** `max_length = 256` tokens.

### Category H: BERT Architecture & Head
36. **Q:** What vector is passed to the linear classification head?  
    **A:** The 768-dimensional `[CLS]` token output vector $\mathbf{h}_{\text{CLS}}$.
37. **Q:** What is the linear head structure?  
    **A:** `Linear(in_features=768, out_features=6)`.
38. **Q:** What activation function is applied to the 6 logits?  
    **A:** Sigmoid activation $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$.
39. **Q:** What dropout rate is applied before the classification head?  
    **A:** Dropout probability of 0.10 (10%).
40. **Q:** How are output logits converted to binary decisions?  
    **A:** By comparing sigmoid probability $p_k$ to category threshold $\tau_k$.

### Category I: Multi-Label & Loss Function
41. **Q:** What is the difference between multi-class and multi-label classification?  
    **A:** Multi-class predicts 1 mutually exclusive class using Softmax; multi-label predicts 0, 1, or multiple classes using Sigmoids.
42. **Q:** Why did you use `BCEWithLogitsLoss`?  
    **A:** It combines Sigmoid activation and Binary Cross-Entropy in a numerically stable PyTorch class.
43. **Q:** What is `pos_weight` in BCE loss?  
    **A:** Positive class weighting factor $w_k = (N - N_k)/N_k$ that penalizes false negatives on positive instances.
44. **Q:** Why is `pos_weight` necessary for safety datasets?  
    **A:** Negative instances outnumber positive instances; `pos_weight` prevents the model from favoring all-zero predictions.
45. **Q:** What is the shape of the target vector?  
    **A:** $6 \times 1$ binary vector `[DV, SH, ST, CA, WH, OV]`.

### Category J: Training & Optimization
46. **Q:** What optimizer was used to train BERT v2?  
    **A:** AdamW optimizer.
47. **Q:** What learning rate was used?  
    **A:** $3 \times 10^{-5}$ (3e-5).
48. **Q:** What weight decay was applied?  
    **A:** $0.01$ weight decay.
49. **Q:** How many epochs was BERT v2 trained for?  
    **A:** 6 Epochs.
50. **Q:** What warmup schedule was used?  
    **A:** Linear warmup over the first 10% of training steps.

### Category K: Threshold Tuning
51. **Q:** Why tune decision thresholds instead of using 0.50?  
    **A:** Different categories have different probability distributions and class frequencies under imbalanced loss.
52. **Q:** Where were thresholds tuned?  
    **A:** Strictly on validation probabilities ($N=162$).
53. **Q:** Was the test set exposed to threshold search?  
    **A:** **No.** Test data was strictly isolated.
54. **Q:** What are the final optimal thresholds?  
    **A:** DV=0.73, SH=0.50, ST=0.79, CA=0.45, WH=0.29, OV=0.63.
55. **Q:** Which category has the lowest threshold and why?  
    **A:** Workplace Harassment (0.29) to compensate for lower positive training frequency.

### Category L: Model Comparison & Results
56. **Q:** What is the Micro F1-score of BERT v2?  
    **A:** 98.09%.
57. **Q:** What is the Exact Match accuracy of BERT v2?  
    **A:** 95.73%.
58. **Q:** What is the Macro F1-score of BERT v2?  
    **A:** 97.58%.
59. **Q:** How many false negatives did BERT v1 produce vs BERT v2?  
    **A:** BERT v1 produced 23 FN; BERT v2 produced 1 FN (95.65% reduction).
60. **Q:** What was the multi-label exact match of BERT v1 vs v2?  
    **A:** BERT v1 = 0.00%; BERT v2 = 94.12%.

### Category M: Error Analysis
61. **Q:** What was the single false negative case in BERT v2?  
    **A:** Incident `TEST_0050` (`[SH, CA]`), where SH probability ($0.4549$) missed its threshold ($0.50$) by $0.0451$.
62. **Q:** How many total false positives did BERT v2 make?  
    **A:** 6 false positives across 984 binary decisions.
63. **Q:** What was the main cause of errors in BERT v1?  
    **A:** Sequence truncation (`max_length=64`), under-training (2 epochs), and unweighted BCE loss.
64. **Q:** Did BERT v2 overfit?  
    **A:** No. High test set metrics (98.09% Micro-F1) confirm strong generalization.
65. **Q:** Where is error analysis documented?  
    **A:** In `ml/evaluation/bert_error_analysis_report.md`.

### Category N: Named Entity Recognition (NER)
66. **Q:** What is the role of NER in ABHERA?  
    **A:** To extract structured evidentiary entities (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`) from narrative text.
67. **Q:** What is the entity-level Micro-F1 score of the NER model?  
    **A:** 99.24%.
68. **Q:** What is the entity-level Micro Precision and Recall?  
    **A:** Precision = 100.00%, Recall = 98.48%.
69. **Q:** How many gold entity spans were in the test evaluation?  
    **A:** 66 gold spans across 60 test records.
70. **Q:** How are character spans aligned to WordPiece tokens?  
    **A:** Using HuggingFace `return_offsets_mapping=True`.

### Category O: Question Engine
71. **Q:** What does the Question Engine do?  
    **A:** Dynamically selects context-aware follow-up questions to fill missing information gaps.
72. **Q:** What are the required entities for Domestic Violence?  
    **A:** `PERP_REL`, `INCIDENT_DETAILS`, `SAFETY_STATUS`.
73. **Q:** What is the priority order for missing requirements?  
    **A:** `SAFETY_STATUS` -> `PERP_REL` -> `LOCATION` -> `TIME_FREQ` -> `PLATFORM` -> `INCIDENT_DETAILS`.
74. **Q:** How does the engine personalize questions?  
    **A:** `generate_dynamic_question()` extracts narrative context phrases (e.g. *"your husband"*) to frame questions.
75. **Q:** How are repeated questions prevented?  
    **A:** Asked question IDs are tracked in `submission_question_answers` and excluded during selection.

### Category P: Session Management & DB
76. **Q:** What are the 4 conversation statuses?  
    **A:** `ACTIVE`, `QUESTIONING`, `PROCESSING`, `COMPLETED`.
77. **Q:** How many database tables exist?  
    **A:** 11 ORM tables.
78. **Q:** What database engine is used?  
    **A:** PostgreSQL managed via SQLAlchemy ORM.
79. **Q:** What is stored in `incident_submissions`?  
    **A:** Initial narrative text, state, district, predicted labels, and extracted entities.
80. **Q:** What is stored in `reports`?  
    **A:** Final compiled 10-section report JSON payloads.

### Category Q: Legal & Support Mapping
81. **Q:** How are laws mapped?  
    **A:** Parameterized SQL queries filter `Law.applicable_label.in_(normalized_codes)`.
82. **Q:** What is the fallback message if no law matches?  
    **A:** `"Information not available in the provided knowledge base."`
83. **Q:** How are support services ranked by location?  
    **A:** District match = 300, State match = 200, National ("All India") = 100.
84. **Q:** What is the fallback message if no support service matches?  
    **A:** `"No matching support service was found in the available support-services database."`
85. **Q:** Where are laws and support services stored?  
    **A:** PostgreSQL `laws` and `support_services` tables.

### Category R: Anti-Hallucination
86. **Q:** How does ABHERA prevent legal hallucinations?  
    **A:** `AntiHallucinationService` validates all legal and support records against PostgreSQL tables verbatim.
87. **Q:** Does BERT generate legal advice text?  
    **A:** **No.** BERT only predicts category codes. Legal text comes directly from PostgreSQL.
88. **Q:** What happens if a legal record text is modified?  
    **A:** It is rejected by `validate_legal_record` with: `"Fabricated legal record rejected"`.
89. **Q:** What is the core anti-hallucination principle?  
    **A:** *"The model predicts and extracts; the database verifies and provides."*
90. **Q:** What audit metrics are produced by the anti-hallucination service?  
    **A:** Received count, accepted count, rejected count, and rejection reasons.

### Category S: Report Generation & PDF
91. **Q:** How many sections exist in the incident report?  
    **A:** 10 conceptual sections.
92. **Q:** Is report generation server-side or client-side?  
    **A:** JSON assembly is server-side (`ReportGeneratorService`); PDF rendering is client-side (`jsPDF` in `Report.jsx`).
93. **Q:** What color tokens are used for the PDF?  
    **A:** `ABHERA_COLORS` (Warm Ivory `#F7F5F0`, Legal Navy `#1E4F8A`, Navy Black `#101820`, Pale Blue `#F4F8FC`, Gold `#B89A5A`).
94. **Q:** How are vector checkmarks drawn in the PDF?  
    **A:** Using vector line primitive calls (`line()`), preventing font encoding errors.
95. **Q:** What disclaimer is attached to reports?  
    **A:** A standard legal disclaimer stating the report provides informational assistance and does not replace qualified legal counsel.

### Category T: System Integration & Audits
96. **Q:** How many unit tests passed in the backend test suite?  
    **A:** 154 / 154 tests passed (100% pass rate).
97. **Q:** Did the end-to-end validation pass?  
    **A:** Yes, all 11 subsystems passed in `abhera_end_to_end_validation.json`.
98. **Q:** How was data leakage audited?  
    **A:** Using MD5 hash comparisons and TF-IDF cosine similarity in `run_leakage_audit.py`.
99. **Q:** How many cross-split hash duplicates were found?  
    **A:** Zero.
100. **Q:** What is the overall documentation status?  
     **A:** `PASS`.

---

# PART 49 — "WHY DID WE USE THIS?" TABLE

| Technology / Method | Why Used in ABHERA | Alternative | Why Not Alternative | Actual Implementation Location |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | Rich ecosystem for deep learning (PyTorch, Transformers) and web APIs (FastAPI). | Java / C++ | Limited NLP library ecosystem for rapid transformer fine-tuning. | Entire repository (`ml/`, `backend/`). |
| **FastAPI** | High-performance asynchronous REST API framework with automatic OpenAPI docs and Pydantic validation. | Flask / Django | Flask lacks native async/Pydantic validation; Django is overly monolithic. | `backend/app/main.py`. |
| **React 18** | Component-based UI rendering, responsive state management, and rich interactive components. | Angular / Vue | React provides lightweight integration with custom Context state. | `frontend/src/App.jsx`. |
| **PostgreSQL** | Enterprise relational database supporting ACID transactions, JSONB storage, and complex SQL filtering. | MongoDB | MongoDB lacks strong relational foreign-key integrity for legal/support mapping. | `backend/app/database.py`. |
| **SQLAlchemy** | Python ORM providing type-safe database queries and parameterized SQL injection prevention. | Raw SQL | Raw SQL strings risk SQL injection vulnerabilities and maintenance friction. | `backend/app/models/*.py`. |
| **`bert-base-uncased`** | Pretrained bidirectional Transformer capturing complex semantic context across 768 dimensions. | BiLSTM / RNN | Recurrent architectures fail to capture long-range bidirectional context effectively. | `ml/training/train_bert_experiment_v2.py`. |
| **WordPiece Tokenizer** | Handles out-of-vocabulary subwords gracefully by breaking unknown words into subwords. | Word-level Split | Word-level splitting creates large sparse vocabularies and handles typos poorly. | `transformers.AutoTokenizer`. |
| **Sigmoid Activation** | Computes independent category probabilities $p_k \in [0, 1]$ for multi-label classification. | Softmax | Softmax forces probabilities to sum to 1.0, failing on compound multi-label incidents. | `ml/inference/predict_bert.py`. |
| **`BCEWithLogitsLoss`** | Combines Sigmoid + BCE with `pos_weight` for numerical stability and class imbalance handling. | MSE / CE | Mean Squared Error fails on classification; Standard CE assumes single-class outputs. | `train_bert_experiment_v2.py`. |
| **AdamW Optimizer** | Decouples weight decay regularization from gradient updates, improving transformer fine-tuning. | Standard SGD | SGD converges slowly on high-dimensional transformer weight landscapes. | `train_bert_experiment_v2.py`. |
| **TF-IDF Vectorizer** | Baseline sparse feature extraction over unigrams/bigrams (`ngram_range=(1,2)`). | Bag-of-Words | Raw word counts ignore word frequency attenuation across documents. | `ml/evaluation/run_baseline_comparison.py`. |
| **Logistic Regression** | Linear baseline establishing benchmark log-odds decision boundaries. | Naive Bayes | Naive Bayes assumes strict feature independence, which fails on n-grams. | `ml/evaluation/baseline_logistic_regression.py`. |
| **Linear SVM** | Maximum-margin linear classification baseline for high-dimensional sparse matrices. | Decision Trees | Un-ensembled decision trees overfit on 5,000-dimensional sparse feature spaces. | `ml/evaluation/baseline_linear_svm.py`. |
| **BERT-NER Token Classifier** | Extracts token-level evidentiary entity spans (`PERP_REL`, `LOCATION`, etc.). | Regex Rules | Rule-based regex fails on natural language narrative variations and typos. | `ml/ner/train_ner.py`. |
| **BIO Schema** | Distinguishes entity boundaries (Begin vs Inside vs Outside) for multi-word entity spans. | Simple Tagging | Simple tagging cannot distinguish adjacent distinct entities of the same class. | `ml/ner/prepare_training_data.py`. |
| **Dynamic Question Engine** | Interactively resolves missing evidentiary requirements in priority order without duplication. | Static Forms | Static forms force survivors to fill out long, triggering, irrelevant questions. | `backend/app/question_engine/engine.py`. |
| **Anti-Hallucination Service** | Verifies legal acts, sections, and helplines against PostgreSQL tables verbatim before output. | Generative LLMs | Commercial LLMs frequently fabricate non-existent legal sections and incorrect phone numbers. | `backend/app/services/anti_hallucination.py`. |
| **jsPDF** | Renders downloadable PDF reports directly inside the survivor's browser client. | Server PDF (WeasyPrint) | Server-side PDF generation increases backend CPU load and transmits PDF files over network. | `frontend/src/pages/Report.jsx`. |

---

# PART 50 — "WHAT IF THEY ASK..." VIVA SECTION

### Direct Defense Answers for Hard Examiner Questions

#### Q1: Why did you use BERT when Logistic Regression already achieved 95.73% Exact Match?
> **Defense:** "While Logistic Regression performed strongly on single-label cases due to explicit surface keywords, **BERT v2 outperformed all models in critical safety metrics: achieving a 98.09% Micro-F1 score, 99.45% Micro-Recall (reducing false negatives from 4 down to 1), and 94.12% Multi-Label Exact Match**. In legal safety applications, minimizing false negatives—ensuring no crime category is missed—is far more critical than surface precision alone."

#### Q2: Why did you tune thresholds on validation data instead of test data?
> **Defense:** "Tuning thresholds on test data causes **data leakage** and produces overly optimistic, ungeneralizable performance claims. By strictly isolating the test set ($N=164$) and optimizing thresholds solely on validation probabilities ($N=162$), we ensured that our reported test set metrics reflect true out-of-sample generalization."

#### Q3: How do you prove your system does not hallucinate legal advice?
> **Defense:** "ABHERA decouples prediction from content generation. The deep learning model only predicts category codes (like `DV` or `CA`). The actual legal text is queried directly from PostgreSQL using parameterized ORM queries. Furthermore, our `AntiHallucinationService` validates every retrieved record against database entries verbatim before report assembly, guaranteeing 100% legal provenance."

#### Q4: Why is `max_features=5000` not considered supervised feature selection?
> **Defense:** "`max_features=5000` in `TfidfVectorizer` ranks terms based purely on overall document frequency ($tf$) across raw narratives, independent of target label correlations $\mathbf{y}$. Supervised feature selection explicitly measures statistical target dependencies (e.g. Chi-Square $\chi^2$ or Mutual Information). Therefore, `max_features` is an unsupervised vocabulary boundary."

---

# PART 51 — COMMANDS

### Confirmed Repository Commands

```bash
# 1. Environment Setup & Dependency Installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Database Initialization & Seeding
python backend/app/db/init_db.py
python backend/app/db/seed_questions.py

# 3. Running Backend API Server (FastAPI / Uvicorn)
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Running Frontend Web App (React / Vite)
cd frontend
npm install
npm run dev

# 5. Executing Backend Unit Test Suite (154 Tests)
cd backend
python -m unittest discover tests

# 6. Training BERT v2 Classification Model
python ml/training/train_bert_experiment_v2.py

# 7. Evaluating Baseline Models (Logistic Regression & Linear SVM)
python ml/evaluation/run_baseline_comparison.py

# 8. Verifying Optimal Decision Thresholds
python ml/evaluation/verify_optimal_thresholds.py

# 9. Training BERT-NER Model
python ml/ner/train_ner.py

# 10. Executing Dataset Leakage Audit
python ml/evaluation/leakage_audit/run_leakage_audit.py

# 11. Running Step 5 End-to-End System Validation
python ml/evaluation/run_e2e_validation.py
```

---

# PART 52 — PROJECT FILE MAP

```
D:\Abhera(Mini)\
├── data/                                 # Raw Dataset & Knowledge Base CSV files
│   ├── womens_safety_dataset.csv         # Master Incident Dataset (1,086 records)
│   ├── laws.csv                          # Statutory Legal Knowledge Base (32 law entries)
│   └── support_services.csv              # Verified Support Services (12 helpline entries)
├── backend/                              # FastAPI Microservice Backend
│   ├── app/
│   │   ├── main.py                       # FastAPI Application Entrypoint & CORS setup
│   │   ├── config.py                     # App Settings & PostgreSQL connection parameters
│   │   ├── database.py                   # SQLAlchemy Engine, SessionLocal, Base & get_db()
│   │   ├── models/                       # SQLAlchemy ORM Data Models (11 Tables)
│   │   │   ├── user.py, session.py, incident_submission.py, conversation_message.py,
│   │   │   ├── prediction.py, entity.py, question.py, law.py, support_service.py, report.py
│   │   ├── services/                     # Business Logic Services
│   │   │   ├── chat_service.py           # Central Intake Pipeline Orchestrator
│   │   │   ├── legal_mapping.py          # PostgreSQL Legal Provision Lookup Engine
│   │   │   ├── support_mapping.py        # Location-Aware Support Helpline Ranking Engine
│   │   │   ├── anti_hallucination.py     # Verbatim Database Verification & Gating Layer
│   │   │   ├── report_generator.py       # 10-Section Incident Report Compiler
│   │   │   └── session_service.py        # Session & Submission Lifecycle Manager
│   │   ├── question_engine/              # Dynamic Questionnaire Orchestrator
│   │   │   ├── engine.py, rules.py, question_selector.py, dynamic_generator.py
│   │   ├── routes/                       # REST API Endpoint Routers
│   │   │   ├── health.py, session.py, chat.py, report.py
│   │   └── db/                           # DB Migration & Seeding Scripts (seed_questions.py)
│   └── tests/                            # Backend Unit Test Suite (154 tests)
├── frontend/                             # React 18 Single-Page Web Frontend (Vite)
│   ├── src/
│   │   ├── App.jsx                       # Main Router & Layout Container
│   │   ├── context/ConversationContext.jsx# React Context State & SessionStorage Sync
│   │   ├── services/api.js               # Axios REST API Client Instance
│   │   ├── pages/                        # Page View Components (Home, Chatbot, Processing, Report)
│   │   └── components/                   # UI Elements (Navbar, ChatInput, ChatMessage, Emblem)
├── ml/                                   # Machine Learning Core Pipelines
│   ├── preprocessing/                    # Dataset Preparation & Splitting (prepare_classification.py)
│   ├── training/                         # BERT Training Scripts (train_bert_experiment_v2.py)
│   ├── inference/                        # Production Inference Predictors (predict_bert.py, predict_ner.py)
│   ├── ner/                              # Named Entity Recognition Preparation & Training (train_ner.py)
│   └── evaluation/                       # Baseline Comparison, Thresholds, Leakage & E2E Validation
└── models/                               # Fine-Tuned Model Weights & Config Artifacts
    ├── bert_multilabel/                  # BERT v1 Baseline Model Directory
    ├── bert_multilabel_experiment_v2/    # Promoted BERT v2 Production Model Directory
    └── ner/                              # BERT-NER Production Model Directory
```

---

# PART 53 — SOURCE TRACEABILITY

### Line-by-Line Claim Traceability Matrix

| Major Project Claim | Source File Path | Line / Block Range | Exact Evidence / Code Snippet | Result Artifact File |
| :--- | :--- | :--- | :--- | :--- |
| **70/15/15 Dataset Split & Seed 42** | [`ml/preprocessing/prepare_classification.py`](file:///d:/Abhera(Mini)/ml/preprocessing/prepare_classification.py#L125-L145) | L125–L145 | `np.random.seed(42)`<br>`clean_df.sample(frac=1.0, random_state=42)` | `ml/preprocessing/train_data.csv`, `val_data.csv`, `test_data.csv` |
| **Absence of Supervised Feature Selection** | Entire Repository (`ml/`, `backend/`) | Whole Repo | 0 occurrences of `SelectKBest`, `chi2`, `mutual_info`, `RFE`, `PCA`. | [`ml/evaluation/ACTUAL_FEATURE_SELECTION_ANALYSIS.md`](file:///d:/Abhera(Mini)/ml/evaluation/ACTUAL_FEATURE_SELECTION_ANALYSIS.md) |
| **Baseline TF-IDF Configuration** | [`ml/evaluation/run_baseline_comparison.py`](file:///d:/Abhera(Mini)/ml/evaluation/run_baseline_comparison.py#L165-L170) | L165–L170 | `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=5000)` | [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L20-L37) |
| **BERT v2 Promoted Training Config** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L220-L240) | L220–L240 | `epochs=6`, `batch_size=32`, `lr=3e-5`, `weight_decay=0.01`, `max_length=128` | `models/bert_multilabel_experiment_v2/training_config.json` |
| **BCEWithLogitsLoss with pos_weight** | [`ml/training/train_bert_experiment_v2.py`](file:///d:/Abhera(Mini)/ml/training/train_bert_experiment_v2.py#L274-L280) | L274–L280 | `criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)` | `models/bert_multilabel_experiment_v2/training_config.json` |
| **Validation Optimal Thresholds** | [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L50-L60) | L50–L60 | `DV: 0.73, SH: 0.50, ST: 0.79, CA: 0.45, WH: 0.29, OV: 0.63` | `models/bert_multilabel_experiment_v2/optimal_thresholds.json` |
| **Final BERT v2 Test Metrics** | [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L54-L79) | L54–L79 | `Micro F1: 0.9809, Exact Match: 0.9573, Macro F1: 0.9758, FN: 1` | [`ml/evaluation/PAPER_RESULTS_SECTION.md`](file:///d:/Abhera(Mini)/ml/evaluation/PAPER_RESULTS_SECTION.md) |
| **NER Entity-Level Performance** | `models/ner/test_metrics.json` | L1–L15 | `Micro Precision: 1.0, Micro Recall: 0.9848, Micro F1: 0.9924` | [`ml/evaluation/FINAL_EVALUATION_RESULTS.json`](file:///d:/Abhera(Mini)/ml/evaluation/FINAL_EVALUATION_RESULTS.json#L231-L238) |
| **Anti-Hallucination Verification** | [`backend/app/services/anti_hallucination.py`](file:///d:/Abhera(Mini)/backend/app/services/anti_hallucination.py#L57-L72) | L57–L72 | `db.query(Law).filter(Law.act_name == act, Law.section_number == section)` | `docs/anti_hallucination.md` |
| **10-Section Report Assembly** | [`backend/app/services/report_generator.py`](file:///d:/Abhera(Mini)/backend/app/services/report_generator.py#L140-L360) | L140–L360 | Compiles 10 sections (`incident_summary` ... `disclaimer`) | PostgreSQL `reports.report_data` column |
| **Backend Unit Test Pass Count** | `backend/tests/` | All Tests | 154 / 154 unit tests passing (100% pass rate) | [`ml/evaluation/abhera_end_to_end_validation.json`](file:///d:/Abhera(Mini)/ml/evaluation/abhera_end_to_end_validation.json#L599-L603) |

---

# PART 54 — OBSERVED DISCREPANCIES

### Documented Project Discrepancies

#### Discrepancy 1: Statutory Laws Count
- **`data/laws.csv` & Active Database:** Contains **32 statutory law rows** (excluding CSV header).
- **Early Prototype Documentation:** Referred to 35 law provisions.
- **Current Project Reality:** The actual active PostgreSQL `laws` seed table contains **32 rows**.

#### Discrepancy 2: Support Services Count
- **`data/support_services.csv` & Active Database:** Contains **12 verified support service rows** (excluding CSV header).
- **Initial Design Notes:** Mentioned 22 or 15 helplines.
- **Current Project Reality:** The actual active PostgreSQL `support_services` seed table contains **12 rows**.

#### Discrepancy 3: Sequence Length Across Pipeline Stages
- **BERT v1 Training:** `max_length = 64`.
- **BERT v2 Training:** `max_length = 128`.
- **Production Inference (`predict_bert.py`):** `max_length = 256`.
- **Explanation:** Shorter sequence lengths were used during training to accelerate epoch iterations, while `max_length = 256` is enforced during production inference to avoid truncating long survivor stories.

#### Discrepancy 4: Category Code Mapping Scope
- **Classification Model:** Operates on 6 target labels (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`).
- **Legal/Support Rules (`rules.py`):** Includes additional category aliases like `FPM` (False Promise of Marriage) and `DOWRY`.
- **Explanation:** In the 6D target matrix, `FPM` and `DOWRY` narratives map to `OV` and `DV`, while rule mapping layers expand these aliases to specific statutory acts (e.g. Dowry Prohibition Act 1961).

---

# PART 55 — FINAL CHEAT SHEET

```
================================================================================
                    ABHERA — ONE-PAGE VIVA CHEAT SHEET
================================================================================

PROJECT:      ABHERA (Adaptive BERT-based Harassment Entity Recognition & Assistance)
PROBLEM:      Survivor legal uncertainty, trauma articulation, and LLM legal hallucinations.
SOLUTION:     Fine-tuned BERT multi-label classification + BERT-NER + Dynamic Question Engine 
              + PostgreSQL anti-hallucination legal & support mapping + Client-side PDF export.

FRONTEND:     React 18 + Vite + Context API (sessionStorage sync) + jsPDF client rendering.
BACKEND:      FastAPI + Pydantic + SQLAlchemy ORM + PostgreSQL database.
DATABASE:     11 Tables (users, sessions, incident_submissions, conversation_messages, 
              predictions, entities, questions, submission_question_answers, laws, 
              support_services, reports).

DATASET:      1,086 narrative records (760 Train / 162 Validation / 164 Test; Seed = 42).
LABELS (6):   DV (Domestic Violence), SH (Sexual Harassment), ST (Stalking), 
              CA (Cyber Abuse), WH (Workplace Harassment), OV (Other Violence).

BERT CLASSIFIER: bert-base-uncased (110M params, 12 layers, 768 hidden dim, 12 heads).
MAX LENGTH:   128 tokens (training) / 256 tokens (production inference).
TRAINING V2:  6 Epochs, Batch Size 32, Learning Rate 3e-5, Weight Decay 0.01, AdamW.
LOSS:         BCEWithLogitsLoss with positive-class weighting (pos_weight).
THRESHOLDS:   Validation tuned: DV=0.73, SH=0.50, ST=0.79, CA=0.45, WH=0.29, OV=0.63.

BERT V2 RESULTS: Micro F1-Score = 98.09% | Exact Match Accuracy = 95.73% | Macro F1 = 97.58%
                 Micro Recall = 99.45% | Multi-Label Exact Match = 94.12%
                 Confusion Matrix: TP=180, FP=6, FN=1, TN=797 (False Negatives reduced 23 -> 1).

NER RESULTS:  Entity-level Micro-F1 = 99.24% | Micro Precision = 100.0% | Micro Recall = 98.48%
              Entity Classes: PERP_REL, LOCATION, TIME_FREQ, PLATFORM (13-tag BIO schema).

BASELINES:    TF-IDF LogReg (Micro-F1 = 97.79%) | TF-IDF Linear SVM (Micro-F1 = 97.27%).
FEATURE SEL:  NO EXPLICIT SUPERVISED FEATURE SELECTION IMPLEMENTED. (max_features=5000 is 
              unsupervised vocabulary limitation).

QUESTION ENG: Priority-ordered dynamic resolution: SAFETY_STATUS -> PERP_REL -> LOCATION 
              -> TIME_FREQ -> PLATFORM -> INCIDENT_DETAILS.
ANTI-HALLUC:  Verbatim PostgreSQL verification layer. "Models predict & extract; DB verifies."
UNIT TESTS:   154 / 154 Passed (100% Pass Rate). End-to-End Validation Status: PASS.
================================================================================
```

---

# PART 56 — FINAL "LEARN THIS IN ORDER"

### 16-Stage Recommended Study Order for Viva Preparation

#### STAGE 1: Dataset & Data Validation (Parts 4 & 5)
- **Study:** 1,086 records, 6 labels (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`), 70/15/15 split, seed 42.
- **Understand:** Validation (`prepare_classification.py`) checks null/empty strings before splitting.

#### STAGE 2: Feature Extraction & Selection Audit (Parts 7 & 8)
- **Study:** TF-IDF parameters (`ngram=(1,2)`, `sublinear_tf=True`, `min_df=2`, `max_features=5000`).
- **Understand:** Why **no explicit supervised feature selection is implemented** (`max_features` is unsupervised).

#### STAGE 3: Classical Baselines (Part 9)
- **Study:** Logistic Regression (97.79% F1) and Linear SVM (97.27% F1) baselines.
- **Understand:** `OneVsRestClassifier` multi-label strategy.

#### STAGE 4: BERT Fundamentals & Tokenization (Parts 10 & 11)
- **Study:** `bert-base-uncased` (110M params, 12 layers, 768 dim), `[CLS]` (101), `[SEP]` (102), `[PAD]` (0).
- **Understand:** Max sequence lengths (128 train / 256 inference) and `attention_mask`.

#### STAGE 5: BERT Architecture & Multi-Label Head (Parts 12 & 13)
- **Study:** $\mathbf{h}_{\text{CLS}} \to \text{Linear}(768 \to 6) \to \text{Sigmoid} \to \text{Thresholds}$.
- **Understand:** Why multi-label uses independent Sigmoids instead of Softmax.

#### STAGE 6: Weighted Loss & Training (Parts 14 & 15)
- **Study:** `BCEWithLogitsLoss` + `pos_weight`, AdamW (lr=3e-5, weight_decay=0.01), 6 epochs.
- **Understand:** How positive loss weighting penalizes false negatives on imbalanced data.

#### STAGE 7: Threshold Tuning (Part 19)
- **Study:** Optimal thresholds (DV=0.73, SH=0.50, ST=0.79, CA=0.45, WH=0.29, OV=0.63).
- **Understand:** Why thresholds were tuned strictly on validation data to prevent test leakage.

#### STAGE 8: BERT v1 vs v2 Evolution (Parts 16, 17, 18, 20 & 21)
- **Study:** Micro-F1 (91.33% -> 98.09%), Exact Match (83.54% -> 95.73%), FN reduction (23 -> 1).
- **Understand:** Why longer max_length, 6 epochs, and weighted loss fixed v1 errors.

#### STAGE 9: Named Entity Recognition (Parts 23 & 24)
- **Study:** BERT-NER model, 13-tag BIO schema, offset mapping subword alignment.
- **Understand:** Entity-level Micro-F1 = 99.24% on 66 gold test spans.

#### STAGE 10: Dynamic Question Engine (Part 25)
- **Study:** Category-to-entity rules, known vs missing info, priority question selection.
- **Understand:** Personalized context phrases and no-repeat tracking.

#### STAGE 11: Database & Session Lifecycle (Parts 26 & 27)
- **Study:** 11 ORM tables (`users`, `sessions`, `incident_submissions`, `reports`, etc.).
- **Understand:** Status lifecycle (`ACTIVE` -> `QUESTIONING` -> `PROCESSING` -> `COMPLETED`).

#### STAGE 12: Legal & Support Mapping (Parts 28 & 29)
- **Study:** Category to `laws` table lookup and location-aware `support_services` ranking (300/200/100).
- **Understand:** Verified fallback messages when no match is found.

#### STAGE 13: Anti-Hallucination Safeguard (Part 30)
- **Study:** `AntiHallucinationService` verbatim PostgreSQL verification.
- **Understand:** Why AI models predict & extract while database verifies & provides.

#### STAGE 14: Report & PDF Generation (Parts 31, 32, 33 & 34)
- **Study:** 10 report sections, FastAPI endpoints, React context, jsPDF client rendering.
- **Understand:** Master design tokens (`ABHERA_COLORS`) and vector checkmark rendering.

#### STAGE 15: Evaluation, Leakage & End-to-End Validation (Parts 35, 36, 37 & 38)
- **Study:** 154 backend unit tests (100% pass rate), 0 leakage hash duplicates, E2E validation.
- **Understand:** Promoted v2 production deployment workflow.

#### STAGE 16: Viva Defense & Cheat Sheet (Parts 48, 49, 50 & 55)
- **Study:** Memorize the 100 viva Q&As, technology justification table, and 1-page cheat sheet.
- **Understand:** Confidently defend model choices, performance metrics, and anti-hallucination architecture.
