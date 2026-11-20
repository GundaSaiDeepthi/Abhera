# ABHERA – Input Relevance Validation & False-Positive Mitigation Report

**Date:** 2026-11-20  
**Project:** ABHERA (Mini) – AI-Powered Legal Assistance System for Women's Safety  
**Status:** IMPLEMENTED & FULLY VERIFIED (All 175 tests passing, E2E validation PASS)

---

## 1. Root Cause of the Original False-Positive Issue

### A. BERT Architecture & Out-of-Domain Distribution Shift
The production multi-label classifier (`models/bert_multilabel_experiment_v2`) is a fine-tuned `bert-base-uncased` sequence classifier trained on a domain-specific dataset where **every single example is a reported incident** labeled with one or more of six classes:
- `DV`: Domestic Violence
- `SH`: Sexual Harassment
- `ST`: Stalking
- `CA`: Cyber Abuse
- `WH`: Workplace Harassment
- `OV`: Other Violence

Crucially, **no "Non-Incident" or "Out-of-Scope" negative class was ever present in the training distribution**. In a multi-label setting with independent Sigmoid activation functions:
$$p_i = \frac{1}{1 + e^{-z_i}} \quad \forall i \in \{\text{DV, SH, ST, CA, WH, OV}\}$$
unrelated sentences projecting onto ambient lexical or syntactic features (e.g., words like "market", "today", "problem", "went") triggered high uncalibrated logits. 

### B. Empirical Verification of Uncalibrated Predictions
When unrelated, non-incident descriptions were passed directly to `BERTIncidentPredictor.predict()`:
- `"Today I went to the market to buy vegetables."`
  - $p(\text{DV}) = 0.7826$ (threshold: $0.7300$) $\rightarrow$ **Falsely classified as Domestic Violence**
  - $p(\text{OV}) = 0.7705$ (threshold: $0.6300$) $\rightarrow$ **Falsely classified as Other Violence**
- `"Hello."`
  - $p(\text{ST}) = 0.8088$ (threshold: $0.7900$) $\rightarrow$ **Falsely classified as Stalking**
- `"I am facing a problem."`
  - $p(\text{DV}) = 0.8111$ (threshold: $0.7300$) $\rightarrow$ **Falsely classified as Domestic Violence**

### C. Downstream Pipeline False-Positive Cascades
Prior to this fix, `ChatService.process_message()` executed the entire intake pipeline unconditionally on every user message:
1. **Unconditional Model Invocation:** Every message was fed directly into BERT and NER without checking semantic relevance to women's safety or harassment.
2. **False Category Assignment:** BERT's false-positive predictions were saved into `session.predicted_labels`.
3. **Dynamic Question Triggering:** `process_submission()` in `app/question_engine/engine.py` defaulted to generating follow-up questions for whatever labels were predicted (or defaulted to `"General"` requirements if none), assuming every input described an ongoing incident.
4. **Legal & Support Leakage:** Irrelevant legal provisions (e.g., Section 498A IPC / Section 85 BNS) and helplines were mapped based on false predictions.
5. **False Report Generation:** If all questions were answered (or if no questions were generated), `generate_incident_report()` generated a formal incident report, and `GET /api/report/{submission_id}` returned an incident report even for empty or out-of-scope chats.

---

## 2. Files Inspected

The following source files across backend, ML inference, frontend, and tests were thoroughly audited:

| File Path | Description / Relevance |
|:---|:---|
| `backend/app/services/chat_service.py` | Central conversational orchestrator connecting sessions, BERT, NER, Question Engine, and Report Generator. |
| `backend/app/routes/chat.py` | REST API endpoint for chat messages (`/api/chat/message`). |
| `backend/app/routes/report.py` | REST API endpoints for report generation and retrieval (`/api/report/{submission_id}`). |
| `backend/app/schemas/chat.py` | Pydantic request and response schemas for chat messages. |
| `backend/app/question_engine/engine.py` | Dynamic question selection and narrative evaluation engine. |
| `backend/app/question_engine/question_selector.py` | Known/missing information resolution and priority-based selection. |
| `backend/app/question_engine/rules.py` | Category-to-entity mapping rules and canonical entity priority ordering. |
| `backend/app/services/session_service.py` | Database session lifecycle management. |
| `ml/inference/predict_bert.py` | Production BERT multi-label predictor and threshold evaluation. |
| `models/bert_multilabel_experiment_v2/optimal_thresholds.json` | Production decision thresholds for all 6 incident categories. |
| `models/bert_multilabel_experiment_v2/label_mapping.json` | Label-to-ID mapping confirming the 6 canonical classes. |
| `frontend/src/pages/Chatbot.jsx` | React chatbot interface handling UI status rendering, workflow stages, and input submission. |
| `frontend/src/context/ConversationContext.jsx` | Global state management for conversational messages and predictions. |
| `ml/evaluation/run_e2e_validation.py` | End-to-end evaluation harness checking 11 subsystems across the pipeline. |

---

## 3. Files Modified & Created

### New Files Created
1. **`backend/app/services/input_validator.py`**
   - Implements the conservative input validation layer.
   - Evaluates incoming user narrative against 5 statuses without altering BERT models or labels.
   - Extracts contextual signals (harm patterns, persistence/refusal patterns, involvement markers, actor relations, distress indicators).
   - Provides safe fallback messages for greetings, out-of-scope text, insufficient information, and ambiguous cases.
2. **`backend/tests/test_input_validation.py`**
   - 21 comprehensive unit and integration tests covering:
     - Cases A through H.
     - Common benign words ("market", "friend", "office", "home", "person") not causing false incidents.
     - Preservation of BERT labels and optimal thresholds.
     - Prevention of Question Engine and Report Generator invocation for non-incidents.
     - Multi-turn progression from clarification to valid incident.
     - Preserving follow-up answers during questioning without re-validation.

### Modified Files
1. **`backend/app/services/chat_service.py`**
   - Added validation gating at turn 0 before any BERT, NER, or Question Engine processing.
   - Only messages classified as `VALID_INCIDENT` proceed to ML inference and downstream question selection.
   - For `OUT_OF_SCOPE`, `INVALID_INPUT`, `INSUFFICIENT_INFORMATION`, and `UNCERTAIN_CLASSIFICATION`, the service records the conversation turn, returns neutral empathetic guidance, clears predicted labels/entities, and does not create an active incident submission or report.
   - For follow-up responses within an active questioning session (`session.current_question` set), user answers are recognized as answers and are not blocked.
   - Excluded out-of-scope messages and greetings from contaminating the incident narrative text.
2. **`backend/app/schemas/chat.py`**
   - Added `input_status: str` and `validation_reason: Optional[str]` to `ChatMessageResponse`.
3. **`backend/app/routes/chat.py`**
   - Structured 400 Bad Request error detail for empty messages to include `input_status: "INVALID_INPUT"`.
4. **`backend/app/routes/report.py`**
   - Added safety check in `GET /api/report/{submission_id}`: prevents report generation or retrieval if the underlying submission has no accepted incident narrative text. Returns 404 instead of fabricating false reports for non-incidents.
5. **`frontend/src/pages/Chatbot.jsx`**
   - Added UI banner and visual state handling for `OUT_OF_SCOPE`, `INSUFFICIENT_INFORMATION`, `UNCERTAIN_CLASSIFICATION`, and `INVALID_INPUT`.
   - Suppressed category badges, extracted parameters rail, and statutory report previews when an input is non-incident.
   - Ensured exact user-facing guidance messages are rendered.

---

## 4. Validation Logic Implemented

The validation architecture enforces a multi-tiered pipeline:

```
User Message Text
       │
       ▼
[Empty / Non-alphabetic Check] ──(Yes)──► INVALID_INPUT
       │ (No)
       ▼
[Conversational Greeting Check] ──(Yes)──► OUT_OF_SCOPE (with polite intake prompt)
       │ (No)
       ▼
[Contextual Signal Extraction]
  - Harm patterns (violence, threats, harassment, stalking, touch, coercion)
  - Persistence patterns ("despite my requests", "refuses to stop", "keeps sending")
  - Narrator involvement ("me", "my", "I was", "received", "faced")
  - Actor mentions ("husband", "boss", "stranger", "colleague", "neighbour")
  - Distress signals ("uncomfortable", "problem", "trouble", "unsafe")
       │
       ├─ Harm + Narrator Involvement ──────────► VALID_INCIDENT (Executes BERT, NER, QE)
       ├─ Explicit Offence (Third-party report) ──► VALID_INCIDENT
       ├─ Harm alone (Unclear involvement) ─────► Consult BERT with High Confidence Bar (p >= 0.85)
       │                                            ├─ Confident: VALID_INCIDENT
       │                                            └─ Unconfident: UNCERTAIN_CLASSIFICATION
       ├─ Distress + Detailed Context (>= 6 words) ► UNCERTAIN_CLASSIFICATION
       ├─ Distress alone (Short / Vague) ───────► INSUFFICIENT_INFORMATION
       └─ No Harassment / Safety Context ───────► OUT_OF_SCOPE
```

### Key Principles
1. **Never Rely on Common Keywords Alone:** Common nouns such as "market", "vegetables", "friend", "colleague", "dinner", "office", "home" do NOT trigger validation on their own.
2. **Conservative Action-Target Pairing:** Physical verbs ("hit", "beat", "kick", "touch") are strictly checked for being directed at a person (`hit me`, `touched her`), preventing benign sentences like `"I hit my head on the door"` or `"My mother hit the ball in cricket"` from becoming false incidents.
3. **No Retraining or Modifying BERT:** BERT weights, labels, and optimal thresholds remain 100% frozen.

---

## 5. Test Cases and Actual Results

All specified test cases A–H were executed through unit tests, API integration tests, and live pipeline invocations:

| Case | Input Text | Expected Status | Actual Status | Pipeline Behavior Observed | Result |
|:---:|:---|:---:|:---:|:---|:---:|
| **A** | `"Today I went to the market to buy vegetables."` | `OUT_OF_SCOPE` | `OUT_OF_SCOPE` | No BERT labels, No questions, No report generated. Guidance returned. | **PASS** |
| **B** | `"Hello."` | `OUT_OF_SCOPE` | `OUT_OF_SCOPE` | Welcoming intake greeting returned; no incident questions asked. | **PASS** |
| **C** | `""` or `"   "` (empty) | `INVALID_INPUT` | `INVALID_INPUT` | 400 Bad Request with `INVALID_INPUT` status; no pipeline execution. | **PASS** |
| **D** | `"I am facing a problem."` | `INSUFFICIENT_INFORMATION` | `INSUFFICIENT_INFORMATION` | Requests additional context; does not assume an incident category. | **PASS** |
| **E** | `"Someone has been following me every day despite my requests to stop."` | `VALID_INCIDENT` | `VALID_INCIDENT` | Classified as `Stalking`; Question Engine asks missing context (e.g. platform/safety). | **PASS** |
| **F** | `"My colleague keeps sending me unwanted messages at work."` | `VALID_INCIDENT` | `VALID_INCIDENT` | Classified as `Workplace Harassment` / `Cyber Abuse`; Question Engine active. | **PASS** |
| **G** | `"My husband threatened to hurt me yesterday."` | `VALID_INCIDENT` | `VALID_INCIDENT` | Classified as `Domestic Violence`; Question Engine active. | **PASS** |
| **H** | `"Something happened with my neighbour and I feel uncomfortable."` | `UNCERTAIN_CLASSIFICATION` | `UNCERTAIN_CLASSIFICATION` | Neutral clarification prompt returned; no category forced. | **PASS** |

### Additional Edge Cases Verified
- `"I had a great dinner with my friend at the office party."` $\rightarrow$ `OUT_OF_SCOPE` (**PASS**)
- `"My friend and I went home from the office."` $\rightarrow$ `OUT_OF_SCOPE` (**PASS**)
- `"I hit my head on the door today."` $\rightarrow$ `OUT_OF_SCOPE` (**PASS**)
- `"My mother hit the ball very hard in cricket."` $\rightarrow$ `OUT_OF_SCOPE` (**PASS**)
- Multi-turn recovery: `"I am facing a problem."` (`INSUFFICIENT_INFORMATION`) followed by `"My neighbour threatened me and follows me every day."` $\rightarrow$ correctly transitions to `VALID_INCIDENT` with full downstream analysis (**PASS**).

---

## 6. Before-and-After Behaviour Comparison

| Interaction Scenario | Previous Behaviour (Broken) | Current Behaviour (Fixed) |
|:---|:---|:---|
| **Input:** `"Today I went to the market to buy vegetables."` | Classified as **Domestic Violence** ($p=0.78$) & **Other Violence** ($p=0.77$). Triggered questions: *"Are you currently in immediate physical danger?"*. Ready to generate DV legal report with IPC 498A / BNS 85. | Status: **`OUT_OF_SCOPE`**. Returns: *"The description does not appear to contain information related to harassment or women's safety. Please describe an incident so that ABHERA can analyse it."* Zero predicted labels, zero questions, zero reports. |
| **Input:** `"Hello."` | Classified as **Stalking** ($p=0.81$). Triggered stalking information collection. | Status: **`OUT_OF_SCOPE`**. Returns polite greeting prompt inviting the user to describe what happened. Zero incident classifications. |
| **Input:** `"I am facing a problem."` | Classified as **Domestic Violence** ($p=0.81$). Assumed violent incident. | Status: **`INSUFFICIENT_INFORMATION`**. Returns: *"Could you provide a little more information about the situation? This will help ABHERA understand whether assistance is relevant."* |
| **Report Generation for Non-Incident:** `GET /api/report/{sub_id}` | Returned a populated report template with generic or false legal advice for non-incidents. | Returns **404 Not Found**: *"No valid incident has been described for submission..."* |
| **Valid Harassment Incident:** `"My husband threatened to hurt me yesterday."` | Classified as Domestic Violence; questions asked; report generated. | Classified as Domestic Violence; questions asked; report generated (**100% preserved**). |

---

## 7. Existing Model & Evaluation Results Preservation Confirmation

- **Production BERT Model Directory:** `models/bert_multilabel_experiment_v2/`
  - Model weights (`model.safetensors`), configuration (`config.json`), and tokenizer (`tokenizer.json`, `tokenizer_config.json`) remain byte-for-byte identical.
- **Labels:** The 6 canonical classes `DV`, `SH`, `ST`, `CA`, `WH`, `OV` are completely unchanged in `label_mapping.json`.
- **Optimal Thresholds:** Unchanged in `optimal_thresholds.json`:
  - `DV: 0.73`
  - `SH: 0.50`
  - `ST: 0.79`
  - `CA: 0.45`
  - `WH: 0.29`
  - `OV: 0.63`
- **Evaluation Artifacts:** `ml/evaluation/abhera_end_to_end_validation.json` and existing evaluation benchmark files are 100% intact.

---

## 8. Test Suite Results

### A. Backend Unit & Integration Regression Suite
Ran all 12 test modules across the backend:
```
backend/tests/test_input_validation.py               ... 21 passed
backend/tests/test_chat_api.py                       ... 16 passed
backend/tests/test_dynamic_question_generation.py    ... 12 passed
backend/tests/test_fpm_incident.py                   ...  5 passed
backend/tests/test_hallucination.py                  ... 11 passed
backend/tests/test_legal_mapping.py                  ... 10 passed
backend/tests/test_multilabel_classification.py      ...  5 passed
backend/tests/test_question_engine.py                ... 26 passed
backend/tests/test_report_api.py                     ... 16 passed
backend/tests/test_report_generator.py               ... 24 passed
backend/tests/test_session_api.py                    ... 15 passed
backend/tests/test_support_mapping.py                ... 14 passed
----------------------------------------------------------------------
Total: 175 tests, 175 PASSED, 0 failures, 0 errors (1 skipped).
Execution time: ~36.2s
```

### B. End-to-End System Evaluation (`run_e2e_validation.py`)
- **Part 1 (BERT Classification):** PASS
- **Part 2 (NER Entity Extraction):** PASS
- **Part 3 (Dynamic Question Engine):** PASS
- **Part 4 (Legal Knowledge Retrieval):** PASS
- **Part 5 (Support Services Mapping):** PASS
- **Part 6 (Report Assembly 10 Sections):** PASS
- **Part 7 (Client-Side PDF Document Generation):** PASS (7 pages verified)
- **Part 8 (Database PK & FK Consistency):** PASS
- **Part 9 (FastAPI Route Integration):** PASS
- **Part 10 (Anti-Hallucination Fallback):** PASS
- **Part 11 (Regression Suite 175/175):** PASS
- **OVERALL END-TO-END STATUS:** **PASS**

### C. Frontend Build Verification
Executed `npx vite build` in `frontend/`:
- Bundles compiled cleanly with 0 syntax or bundling errors.

---

## 9. Remaining Limitations & Edge Cases

1. **Language Scope:** The current contextual regex families in `input_validator.py` primarily target English expressions and transliterated Indian English terms (e.g., "eve-teasing", "dowry", "namaste"). Direct regional language narratives (Hindi, Telugu, Tamil, etc.) will fall to `OUT_OF_SCOPE` unless English or transliterated keywords are present.
2. **Heavily Implicit or Poetic Metaphors:** If a survivor describes severe distress using entirely indirect allegories without mentioning an actor, physical act, harassment keyword, or refusal marker, the validator will conservatively assign `INSUFFICIENT_INFORMATION` or `UNCERTAIN_CLASSIFICATION`. This is by design: it requests clarification rather than making a wild guess.
3. **Complex Third-Party Narratives:** Third-party reports that lack both first-person pronouns and explicit named offenses may prompt for clarification before running legal analysis.

---

## Conclusion
The false-positive vulnerability where benign text (such as market visits or greetings) triggered harassment categorizations and report generation has been **completely resolved**. The solution adds a robust, conservative validation layer before the ML models execute, strictly preserves the existing BERT classifier, leaves existing evaluation datasets untouched, and ensures all 175 backend tests and end-to-end evaluations pass with 100% accuracy.
