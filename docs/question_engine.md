# Dynamic Question Engine Architecture & Technical Specification

## 1. Overview & Purpose

The **Dynamic Question Engine** replaces static, fixed-sequence questionnaires with a dynamic, state-aware information gathering workflow. 

Rather than asking every user the exact same list of questions, the engine:
1. Identifies the active incident categories (via BERT multi-label classifier).
2. Determines what information is already known (via initial narrative text, NER-extracted entity spans, and previous user answers).
3. Computes the missing required information dimensions for the detected incident categories.
4. Deterministically selects the highest-priority relevant unanswered question from the database.
5. Manages and persists conversation state per submission ID.
6. Returns `completion_status = true` as soon as all required information dimensions are satisfied.

---

## 2. Component Architecture

The module is located under `backend/app/question_engine/`:

```
backend/app/question_engine/
├── __init__.py          # Package exports
├── rules.py             # Category rules, normalization & multi-label combination
├── question_selector.py # Known/missing information logic & question selection algorithm
└── engine.py            # Main orchestrator & state manager
```

---

## 3. Database Schema

The engine relies on two PostgreSQL database tables registered via SQLAlchemy in `app/models/question.py`:

### `questions` (Master Question Bank)
| Column | Type | Constraints / Description |
| :--- | :--- | :--- |
| `question_id` | `VARCHAR(64)` | Primary Key (e.g., `"Q_PERP_REL_01"`) |
| `category` | `VARCHAR(100)` | Target incident category or `"General"` / `"ALL"` |
| `question_text` | `TEXT` | Factual question presented to the user |
| `required_entity` | `VARCHAR(100)` | Entity dimension satisfied by this question (e.g., `"PERP_REL"`, `"LOCATION"`) |
| `priority` | `INTEGER` | Priority rank (higher integer = higher priority) |
| `active` | `BOOLEAN` | Active status flag (default `true`) |

### `submission_question_answers` (State & Answer Tracking)
| Column | Type | Constraints / Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment |
| `submission_id` | `VARCHAR(64)` | Foreign Key -> `incident_submissions.submission_id` (CASCADE) |
| `question_id` | `VARCHAR(64)` | Foreign Key -> `questions.question_id` (CASCADE) |
| `answer` | `TEXT` | User-provided answer text |
| `answered_at` | `TIMESTAMP WITH TIME ZONE` | Timestamp when user answered |
| `asked_at` | `TIMESTAMP WITH TIME ZONE` | Timestamp when question was selected/asked |

---

## 4. Question Selection Algorithm

Selection proceeds deterministically through the following pipeline:

```
BERT Classification Results + NER Entity Spans + Previous Answers
                              ↓
              Compute Required Information Dimensions
                              ↓
             Determine Information Already Known
                              ↓
            missing_info = required_info - known_info
                              ↓
       Is missing_info empty OR no questions remaining?
                    │                      │
                   YES                     NO
                    ↓                      ↓
          completion_status = true  Filter Questions Bank:
                                    - Exclude asked question_ids
                                    - Filter to active == true
                                    - Target entity in missing_info
                                    - Match active category / General
                                    - Sort by Priority & Missing Rank
                                           ↓
                                    Return top question & completion_status = false
```

### Deterministic Score Formula
Candidate questions are ranked by:
$$\text{Score} = \text{priority} + \text{category\_relevance\_boost} + \text{missing\_rank\_score}$$
- `category_relevance_boost`: +20 if category matches active incident category, +10 for General.
- `missing_rank_score`: Higher boost for dimensions listed earlier in the canonical requirement priority order.

---

## 5. State Management & Resumability

Conversation state is bound to `submission_id`:
- **Persistence**: Answers are saved to `submission_question_answers`.
- **Resumability**: When a user leaves and returns to `submission_id`, `process_submission()` fetches all previously asked `question_id`s and answers from PostgreSQL.
- **Deduplication**: Once a question ID is recorded for a submission ID, it is never asked again.

---

## 6. Entity-Dependent Question Suppression

Questions are suppressed dynamically if the required information is already detected in the initial narrative by the supervised NER model:

- If NER extracts `PERP_REL = "former colleague"`:
  - `PERP_REL` is marked as **KNOWN**.
  - `PERP_REL` is removed from `missing_information`.
  - Question `Q_PERP_REL_01` (*"What is your relationship to the person?"*) is **SUPPRESSED**.

- If NER extracts `LOCATION = "workplace"`:
  - `LOCATION` question `Q_LOCATION_01` is **SUPPRESSED**.

- If NER extracts `TIME_FREQ = "every evening"`:
  - `TIME_FREQ` question `Q_TIME_FREQ_01` is **SUPPRESSED**.

- If NER extracts `PLATFORM = "Instagram"`:
  - `PLATFORM` question `Q_PLATFORM_01` is **SUPPRESSED**.

---

## 7. Multi-Label Incident Handling

When BERT predicts multiple incident categories (e.g., `["Domestic Violence", "Stalking"]`):
1. `rules.get_required_information()` fetches requirement lists for all active categories:
   - `Domestic Violence`: `["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"]`
   - `Stalking`: `["PERP_REL", "TIME_FREQ", "PLATFORM", "SAFETY_STATUS"]`
2. The engine computes the **set union** of requirements:
   - Combined Required Set: `["PERP_REL", "SAFETY_STATUS", "TIME_FREQ", "PLATFORM", "INCIDENT_DETAILS"]`
3. A single combined, prioritized question flow is executed rather than running duplicate questionnaires.

---

## 8. Completion Criteria

`completion_status = true` is returned when:
1. `missing_information` is empty (`[]`), meaning all required entity dimensions for the active categories are satisfied by initial narrative, NER entities, or previous answers.
2. OR no remaining active unanswered questions exist for the missing entity dimensions.

---

## 9. Example Question Flows

### Scenario A: Domestic Violence (No Initial NER Entities)
1. **Input**: `submission_id = "SUB-001"`, `BERT = ["Domestic Violence"]`, `NER = []`.
2. **Required**: `["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"]`.
3. **Known**: `{}`.
4. **Missing**: `["SAFETY_STATUS", "PERP_REL", "INCIDENT_DETAILS"]`.
5. **Output**:
   ```json
   {
       "submission_id": "SUB-001",
       "missing_information": ["SAFETY_STATUS", "PERP_REL", "INCIDENT_DETAILS"],
       "next_question": "Are you currently in immediate physical danger, or do you need emergency support right now?",
       "question_id": "Q_SAFETY_01",
       "completion_status": false
   }
   ```

### Scenario B: Stalking with Extracted NER Entities
1. **Input**: `submission_id = "SUB-002"`, `BERT = ["Stalking"]`, `NER = [PERP_REL: "ex-boyfriend", TIME_FREQ: "every day"]`.
2. **Required**: `["PERP_REL", "TIME_FREQ", "PLATFORM", "SAFETY_STATUS"]`.
3. **Known**: `{"PERP_REL": "ex-boyfriend", "TIME_FREQ": "every day"}`.
4. **Missing**: `["SAFETY_STATUS", "PLATFORM"]` (*PERP_REL and TIME_FREQ suppressed*).
5. **Output**:
   ```json
   {
       "submission_id": "SUB-002",
       "missing_information": ["SAFETY_STATUS", "PLATFORM"],
       "next_question": "Which platform, app, or communication channel was used during this incident (e.g., WhatsApp, Instagram, email, phone calls)?",
       "question_id": "Q_PLATFORM_01",
       "completion_status": false
   }
   ```

### Scenario C: Multi-Label Incident (Sexual Harassment + Workplace Harassment)
1. **Input**: `submission_id = "SUB-003"`, `BERT = ["SH", "WH"]`, `NER = [LOCATION: "office pantry"]`.
2. **Required**: `["PERP_REL", "LOCATION", "INCIDENT_DETAILS"]`.
3. **Known**: `{"LOCATION": "office pantry"}`.
4. **Missing**: `["PERP_REL", "INCIDENT_DETAILS"]`.
5. **Output**:
   ```json
   {
       "submission_id": "SUB-003",
       "missing_information": ["PERP_REL", "INCIDENT_DETAILS"],
       "next_question": "What is your relationship to the person involved (e.g., spouse/partner, family member, landlord, colleague, acquaintance, or stranger)?",
       "question_id": "Q_PERP_REL_01",
       "completion_status": false
   }
   ```
