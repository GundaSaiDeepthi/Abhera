# Named Entity Recognition (NER) Annotation Guidelines

> **IMPORTANT NOTICE FOR ANNOTATORS**:
> The examples provided in this document are **illustrative guidelines only** for human annotators.
> They are **NOT** training records and must **NOT** be inserted into the training or validation datasets.

---

## 1. Overview & Dataset Context
The incident dataset (`data/womens_safety_dataset.csv`) contains natural language narratives categorized into multi-label safety classifications. 

However, **it does not contain token-level or span-level NER entity annotations**. In compliance with strict zero-hallucination and zero-fabrication rules, BIO tags and entity spans cannot be automatically generated or guessed by LLMs/heuristics. Ground-truth NER annotations must be produced by human annotators following these guidelines.

---

## 2. Proposed Entity Schema Specifications & Annotation Rules

Annotators will extract the following six entity classes from incident narratives:

### 2.1 `PERP_REL` — Perpetrator Relation
- **Definition**: The relational role or status of the alleged perpetrator relative to the victim.
- **What to annotate**: Words or phrases explicitly stating who the perpetrator is.
- **Allowed Examples (Illustrative Only)**:
  - `"husband"`, `"ex-boyfriend"`, `"senior manager"`, `"brother-in-law"`, `"landlord"`, `"stranger"`, `"community members"`, `"supervisor"`
- **Negative Rules**: Do NOT annotate personal pronouns like `"he"` or `"him"` unless no noun phrase exists.

### 2.2 `LOCATION` — Harassment Location
- **Definition**: Physical, organizational, or virtual setting where the incident occurred.
- **What to annotate**: Specific locations mentioned in text.
- **Allowed Examples (Illustrative Only)**:
  - `"office"`, `"home"`, `"bedroom"`, `"bus stand"`, `"college gate"`, `"parking lot"`, `"in front of guests"`, `"online"`

### 2.3 `TIME_FREQ` — Time & Frequency
- **Definition**: Specific time of day, date, duration, or recurrence pattern of harassment.
- **What to annotate**: Temporal expressions and recurrence frequencies.
- **Allowed Examples (Illustrative Only)**:
  - `"every morning"`, `"after hours"`, `"daily"`, `"for over a year"`, `"last night"`, `"two weeks"`, `"every time they visit"`

### 2.4 `PLATFORM` — Digital Platform & Communication Medium
- **Definition**: Online applications, websites, or communication channels used to harass.
- **What to annotate**: Names of social platforms, messaging channels, or digital media.
- **Allowed Examples (Illustrative Only)**:
  - `"social media"`, `"fake profile"`, `"email"`, `"WhatsApp"`, `"direct messages"`, `"phone calls"`, `"online"`

### 2.5 `EVIDENCE` — Physical Harm, Weapons & Tangible Evidence
- **Definition**: Physical harm actions, weapons used, or tangible physical evidence left behind.
- **What to annotate**: Specific acts of physical violence, weapons, or physical artifacts.
- **Allowed Examples (Illustrative Only)**:
  - `"slapped"`, `"belt"`, `"locked inside"`, `"notes on car"`, `"edited photos"`, `"property dispute"`

### 2.6 `LAW_SEC` — Statutory Law Section Mentions
- **Definition**: Statutory law sections, acts, or legal code numbers mentioned explicitly in text.
- **What to annotate**: Statutory section numbers or legal act names.
- **Allowed Examples (Illustrative Only)**:
  - `"354 IPC"`, `"Section 498A"`, `"74 BNS"`, `"POSH Act"`, `"IT Act"`

---

## 3. Annotation Execution Rules

### 3.1 Character Offsets (`start` & `end`)
- Annotations must record exact character index bounds `[start, end]`.
- `start` = 0-indexed position of first character.
- `end` = 0-indexed position immediately after the last character (`text[start:end]`).

### 3.2 Exact Text Matching
- `ent["text"]` must match `text[start:end]` character-for-character.

### 3.3 No Overlapping Spans
- Overlapping character spans are invalid for standard sequence tagging.
- If two entity concepts overlap, annotate the longer, more informative complete phrase.

### 3.4 Handling Unmentioned Entities
- If an entity category is not mentioned in a narrative, leave it unannotated. Do **not** invent missing entities.

---

## 4. Annotation Quality Assurance & Validation
All completed `.jsonl` files must pass validation via `python ml/ner/annotation/validate_annotations.py` before any future NER model training can occur.
