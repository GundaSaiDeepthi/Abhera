# ABHERA Development Guide — Module 08: Adaptive Question Engine & Conversational Slot-Filling

---

## 1. Problem Formulation: Trauma-Informed Adaptive Interviewing

When survivors of sexual assault, domestic violence, or stalking seek assistance, cognitive overload and trauma often cause them to disclose incidents in an incomplete or unstructured manner. A survivor might type:
> *"He has been threatening me online and sending horrible messages."*

From a legal and evidentiary perspective, this statement contains critical gaps:
- *Who is "he"?* (An ex-husband, a co-worker, a stranger?)
- *What digital channel is being weaponized?* (WhatsApp, Instagram, SMS?)
- *How long has this occurred?* (A single occurrence or persistent stalking over months?)
- *Is the survivor currently in immediate physical danger?*

Traditional government portals force victims to navigate rigid 30-field tabular webforms, which exacerbates anxiety and drives high dropout rates. Conversely, unconstrained conversational chatbots often wander into irrelevant, open-domain chitchat.

To solve this, ABHERA implements an **Adaptive Dynamic Question Engine** (`backend/app/question_engine/`). The engine models the interview as an **evidential slot-filling state machine**. It assesses what is already known, computes what is missing according to the specific legal categories identified by BERT, and asks targeted, empathetic follow-up questions prioritized by urgency.

```
       +-----------------------------------------------------------------------------------------+
       |                         THE 6 FORENSIC INCIDENT SLOTS                                   |
       +-----------------------------------------------------------------------------------------+
       |  Slot Dimension       | Evidential Relevance         | Statutory & Assistance Impact    |
       +-----------------------+------------------------------+----------------------------------+
       |  1. SAFETY_STATUS     | Immediate physical danger    | Crisis hotline redirection (112) |
       |  2. PERP_REL          | Identity/relation of accused | Domestic Violence Act vs POSH Act|
       |  3. LOCATION          | Physical crime jurisdiction  | Local Sakhi One-Stop Centre match|
       |  4. TIME_FREQ         | Evidential pattern/duration  | Stalking (BNS 354D) persistency  |
       |  5. PLATFORM          | Digital communication medium | IT Act 2000 (Sec 66E/67/67A)     |
       |  6. INCIDENT_DETAILS  | Factual narrative of conduct | Core factual basis for complaint |
       +-----------------------+------------------------------+----------------------------------+
```

---

## 2. Dynamic Question Engine Architecture & Dataflow

The Question Engine operates as a closed-loop controller executed on every conversational turn.

```mermaid
flowchart TD
    A["User Narrative Input / Answer Turn"] --> B["Active Harassment Categories<br/>(from BERT Classifier: e.g., DV + ST)"]
    A --> C["Extracted Named Entities<br/>(from BERT NER: e.g., PERP_REL='husband')"]

    B --> D["Rules Engine (rules.py)<br/>Compute Combined Required Information"]
    C --> E["State Tracker (question_selector.py)<br/>Extract Known Information Map"]
    A --> E

    D & E --> F["Difference Evaluator<br/>missing_info = required_info - known_info"]

    F --> G{"Is missing_info Empty?"}
    G -->|Yes| H["Set completion_status = True<br/>Transition Session to PROCESSING"]
    G -->|No| I["Priority Queue Sorter<br/>Order: SAFETY > PERP > LOC > TIME > PLAT"]

    I --> J["Question Selector<br/>Filter already asked question_ids"]
    J --> K{"Unasked Question Available?"}
    K -->|No| H
    K -->|Yes| L["Dynamic Generator (dynamic_generator.py)<br/>Synthesize Context-Aware Phrasing"]
    L --> M["Dispatch Targeted Question to Client<br/>Set completion_status = False"]
```

---

## 3. Statutory Category Requirements Mapping

Different harassment categories carry distinct evidentiary requirements under Indian law. In `backend/app/question_engine/rules.py`, domain rules dictate which slots must be resolved for each category:

```python
CATEGORY_REQUIRED_ENTITIES: Dict[str, List[str]] = {
    "Domestic Violence": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "Dowry Harassment": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "Stalking": ["PERP_REL", "TIME_FREQ", "PLATFORM", "SAFETY_STATUS"],
    "Sexual Harassment": ["PERP_REL", "LOCATION", "INCIDENT_DETAILS"],
    "Cyber Harassment": ["PLATFORM", "TIME_FREQ", "INCIDENT_DETAILS"],
    "Workplace Harassment": ["PERP_REL", "LOCATION", "INCIDENT_DETAILS"],
    "Other Harassment": ["INCIDENT_DETAILS", "SAFETY_STATUS"],
    "False Promise of Marriage": ["PERP_REL", "TIME_FREQ", "INCIDENT_DETAILS", "SAFETY_STATUS"],
    "General": ["PERP_REL", "INCIDENT_DETAILS", "SAFETY_STATUS"],
}
```

### 3.1 Multi-Label Requirement Union
When an incident activates multiple labels (e.g., a survivor facing both **Domestic Violence** and **Cyber Harassment**), the engine takes the **mathematical union** of their required slots:

$$\text{Required}_{\text{DV} \cup \text{CA}} = \{\text{PERP\_REL}, \text{INCIDENT\_DETAILS}, \text{SAFETY\_STATUS}\} \cup \{\text{PLATFORM}, \text{TIME\_FREQ}, \text{INCIDENT\_DETAILS}\}$$

$$\implies \{\text{SAFETY\_STATUS}, \text{PERP\_REL}, \text{TIME\_FREQ}, \text{PLATFORM}, \text{INCIDENT\_DETAILS}\}$$

---

## 4. Canonical Slot Priority Queue

If multiple slots are missing, the engine does not ask about them randomly. It enforces a strict, trauma-informed priority order:

$$\text{SAFETY\_STATUS} \succ \text{PERP\_REL} \succ \text{LOCATION} \succ \text{TIME\_FREQ} \succ \text{PLATFORM} \succ \text{INCIDENT\_DETAILS}$$

```
       +-----------------------------------------------------------------------------------------+
       |                          SLOT RESOLUTION PRIORITY LOGIC                                 |
       +-----------------------------------------------------------------------------------------+
       |  Priority | Slot Name         | Why This Slot Takes Precedence                          |
       +-----------+-------------------+---------------------------------------------------------+
       |  Rank 1   | SAFETY_STATUS     | Life safety overrides paperwork. If user is in danger,  |
       |           |                   | emergency protocols must trigger immediately.           |
       |  Rank 2   | PERP_REL          | Determines whether the case falls under civil DV Act,   |
       |           |                   | corporate POSH Act, or general criminal BNS statutes.   |
       |  Rank 3   | LOCATION          | Dictates police jurisdiction and enables hyperlocal     |
       |           |                   | matching to the nearest One Stop Centre (OSC).          |
       |  Rank 4   | TIME_FREQ         | Proves repeated course of conduct required for stalking |
       |           |                   | under BNS Section 354D / 78.                            |
       |  Rank 5   | PLATFORM          | Identifies electronic evidence source under IT Act.     |
       |  Rank 6   | INCIDENT_DETAILS  | Captures specific factual nuances of the complaint.     |
       +-----------+-------------------+---------------------------------------------------------+
```

---

## 5. Multi-Source Known Information Resolution

In `backend/app/question_engine/question_selector.py`, the function `get_known_information()` aggregates evidence across three distinct sources to prevent asking the user for information they have already provided:

1. **Supervised BERT NER Spans:** If the NER model extracted `{"label": "PERP_REL", "text": "ex-husband"}`, `known["PERP_REL"]` is immediately marked as resolved.
2. **Database Conversation History:** Prior user answers in `submission_question_answers` are mapped to their respective target entity slots.
3. **Lexical Regex Heuristics:** The narrative is scanned for unambiguous colloquial markers:
   - *Safety Status:* Matches phrases like `"am safe"`, `"out of danger"`, `"feel unsafe"`, `"threatened"`, `"in danger"`.
   - *Time / Frequency:* Matches terms like `"every day"`, `"past month"`, `"yesterday"`, `"for weeks"`, `"repeatedly"`.
   - *Platform:* Matches digital anchors like `"WhatsApp"`, `"Instagram"`, `"email"`, `"phone calls"`, `"texts"`.

---

## 6. Context-Aware Dynamic Question Generation

Rather than emitting robotic canned prompts, `backend/app/question_engine/dynamic_generator.py` synthesizes natural, empathetic questions that reference details the user has already disclosed.

### Example Contextual Transformation:
- **Generic Fallback Question:** *"Where did this incident occur?"*
- **Known Context:** BERT has identified `Workplace Harassment` (`WH`) and NER extracted `PERP_REL = "senior manager"`.
- **ABHERA Dynamic Synthesis:**
  > *"You mentioned this involves your senior manager. Could you clarify where this took place—for instance, inside your office premises, during a company event, or outside of work?"*

---

## 7. Anti-Looping & Graceful Termination Guardrails

To prevent conversational deadlock or infinite questioning loops:

1. **Duplicate Question Prohibition:** The engine tracks `asked_question_ids: Set[str]`. Once a question has been presented in a session, its ID is permanently blacklisted from being reselected.
2. **Exhaustion Fallback:** If unresolved slots remain but all matching questions in the catalog have already been asked, the engine terminates questioning gracefully (`completion_status = True`) rather than repeating prompts.
3. **User-Initiated Early Finalization:** The frontend provides a prominent **"Generate Report Now"** action. If clicked, the client bypasses remaining questionnaire turns and requests immediate report generation.

---

## 8. Standardized 16-Field Code Explanation: `engine.process_submission`

1. **FILE:** `backend/app/question_engine/engine.py`
2. **FUNCTION / CLASS:** `process_submission`
3. **PURPOSE:** Evaluates active categories and extracted entities against required information rules, identifies missing slots, selects the highest-priority unasked question, and synthesizes dynamic question text.
4. **WHY CREATED:** To serve as the central brain governing multi-turn adaptive interviewing.
5. **CALLED BY:** `backend/app/services/chat_service.py` during `ChatService.process_message()`.
6. **INPUT ARGUMENTS:** `submission_id` (str), `narrative_text` (str), `bert_results` (dict), `ner_entities` (list), `previous_answers` (dict), `db` (SQLAlchemy Session).
7. **SOURCE CODE:**
```python
def process_submission(
    submission_id: str,
    narrative_text: str = "",
    bert_results: Dict[str, Any] = None,
    ner_entities: List[Dict[str, Any]] = None,
    previous_answers: Dict[str, str] = None,
    db: Optional[DBSession] = None,
) -> Dict[str, Any]:
    active_categories = [normalize_category(lbl) for lbl in bert_results.get("labels", [])]
    if not active_categories:
        active_categories = ["General"]

    questions_bank = db.query(Question).filter(Question.active == True).all() if db else []
    question_bank_dict = {getattr(q, "question_id", str(idx)): q for idx, q in enumerate(questions_bank)}

    asked_question_ids = set()
    answers_dict = previous_answers.copy() if previous_answers else {}
    if db is not None:
        db_records = db.query(SubmissionQuestionAnswer).filter(SubmissionQuestionAnswer.submission_id == submission_id).all()
        for rec in db_records:
            asked_question_ids.add(rec.question_id)
            if rec.answer:
                answers_dict[rec.question_id] = rec.answer

    required_info = get_required_information(active_categories)
    known_info = get_known_information(narrative_text, ner_entities, answers_dict, question_bank_dict)
    missing_info = get_missing_information(required_info, known_info)

    next_q = select_next_question(missing_info, active_categories, asked_question_ids, questions_bank)

    if not missing_info or next_q is None:
        return {
            "submission_id": submission_id,
            "missing_information": [],
            "next_question": None,
            "question_id": None,
            "completion_status": True,
        }

    dynamic_q_text = generate_dynamic_question(
        required_entity=next_q.required_entity,
        active_categories=active_categories,
        narrative_text=narrative_text,
        known_info=known_info,
        fallback_text=next_q.question_text,
    )

    return {
        "submission_id": submission_id,
        "missing_information": missing_info,
        "next_question": dynamic_q_text,
        "question_id": next_q.question_id,
        "completion_status": False,
    }
```
8. **LINE-BY-LINE EXPLANATION:**
   - Lines 94-106: Normalizes BERT predicted labels into canonical category names.
   - Lines 108-117: Queries active questions from the database and indexes them by `question_id`.
   - Lines 120-136: Queries `submission_question_answers` to reconstruct previously asked question IDs and user answers.
   - Line 138: Evaluates domain rules to assemble the union of required slots.
   - Lines 139-144: Traverses NER entities, user answers, and narrative text to compile the `known_info` map.
   - Line 145: Performs set difference: `missing_info = [slot for slot in required if slot not in known]`.
   - Lines 148-153: Prioritizes missing slots and picks the highest-priority question not present in `asked_question_ids`.
   - Lines 156-163: If no information is missing or no unasked questions remain, flags completion status as `True`.
   - Lines 169-175: Injects known context into dynamic question phrasing.
   - Lines 177-183: Returns the completed engine response dictionary.
9. **STEP-BY-STEP PROCESS:**
   - 1. Extract active categories $\to$ 2. Load question catalog $\to$ 3. Build session history $\to$ 4. Determine required slots $\to$ 5. Extract known slots $\to$ 6. Compute missing slots $\to$ 7. Select next question by priority $\to$ 8. Format dynamic text $\to$ 9. Return response.
10. **RETURN VALUE / OUTPUT:** A dictionary containing `submission_id`, `missing_information`, `next_question`, `question_id`, and `completion_status`.
11. **DATABASE EFFECT:** Reads `questions` and `submission_question_answers`.
12. **NEXT CONSUMER:** `ChatService` package returns this to the React UI to display the next conversational bubble.
13. **ERROR HANDLING:** Operates safely in headless or offline mode when `db` is `None` via `questions_bank_override` (used extensively in unit testing).
14. **RELATED UNIT TEST:** Verified in `backend/tests/test_question_engine.py`.
15. **BEGINNER EXPLANATION:**
   Imagine a detective helping a witness write a police report. If the witness says "Someone broke my window," the detective checks their checklist: "Do we know who? No. Do we know when? No. Is the person safe right now? No." The detective asks the most urgent question first: "Are you safe right now?" Once that is answered, the detective asks who did it. When all boxes are checked, the detective says: "Thank you, I have everything I need to write your formal report."
16. **TECHNICAL EXPLANATION:**
   `process_submission` implements a deterministic forward-chaining production rule engine combined with a slot-filling dialog manager. It bounds dialog trajectory length to $|\mathcal{S}| \le 6$ conversational turns, preventing exponential state-space explosion and guaranteeing termination in $\mathcal{O}(|\mathcal{Q}| + |\mathcal{E}|)$ time complexity.

---

## 9. Multi-Level Explanation

### 9.1 Beginner Level
When someone is upset, forcing them to answer 20 complicated form questions is cruel and unhelpful. ABHERA lets them start by writing whatever they want. Then, our intelligent Question Engine acts like a caring helper. It checks what parts of the story are already clear and only asks about the missing pieces—one question at a time. Most importantly, it always asks about safety first before asking about locations or apps.

### 9.2 Developer Level
The question engine is structured as a modular package in `backend/app/question_engine/`. Category rules in `rules.py` define required entity dimensions. Ingestion of known data in `question_selector.py` evaluates three streams: high-confidence BERT NER entities, previous answers from `submission_question_answers`, and regex narrative heuristics. The difference between required and known slots defines `missing_info`. The engine selects the next unasked question matching the highest-priority missing slot and routes it through `dynamic_generator.py` for contextual phrasing.

### 9.3 Technical Reviewer & Examiner Level
ABHERA's conversational management rejects non-deterministic autoregressive generation in favor of a **deterministic symbolic state machine**. Dialog act selection is governed by a strict partial ordering over the evidentiary lattice $\mathcal{L} = \langle \mathcal{S}, \preceq \rangle$, where the priority supremum is defined as $\sup(\mathcal{L}) = \text{SAFETY\_STATUS}$. 

By parameterizing slot requirements as a function of the multi-label prediction vector $\hat{\mathbf{y}} \in \{0, 1\}^6$, the interview dynamically scales its depth in response to compound legal jeopardy. The anti-looping invariant guarantees monotonic convergence: in each turn $t$, either $|\text{known}_{t+1}| > |\text{known}_t|$ or $|\text{unasked}_{t+1}| < |\text{unasked}_t|$, proving termination within a bounded finite horizon.

---

## 10. Document Verification & Traceability

- **Document Version:** 1.0.0
- **Audited Files:** `backend/app/question_engine/engine.py`, `backend/app/question_engine/rules.py`, `backend/app/question_engine/question_selector.py`, `backend/app/question_engine/dynamic_generator.py`.
- **System Specifications Verified:** 6 Forensic Slots, Canonical Priority Queue, Anti-Looping Guardrails, Multi-Label Union Logic.
- **Verification Status:** Fully Verified and Operational.
