# ABHERA — ACTUAL LEGAL INFORMATION AND SUPPORT SERVICE RECOMMENDATION AUDIT

**Document Version:** 1.0.0  
**Project:** ABHERA — AI-Driven Legal Guidance and Multi-Label Incident Classification System for Survivor Support  
**Scope:** Source-Code Inspection, Legal & Support Retrieval Architecture, Anti-Hallucination Gating, and Knowledge Base Audit  
**Status:** Audit Complete & Verified  

---

## 1. PURPOSE

This document provides a comprehensive, code-authoritative audit of how the **ABHERA** platform retrieves, filters, validates, and presents legal information and support services.

In strict compliance with academic paper guidelines and repository audit constraints:
* **No code, database records, datasets, models, or fallback messages have been altered.**
* **All retrieval logic, scoring formulas, dataset schemas, and anti-hallucination rules are documented directly from source code.**
* **No generative AI models generate statutory text or helpline contact details.**
* The core design principle is strictly maintained:  
  *"The model predicts and extracts; the database verifies and provides the final legal and support information."*

---

# 2. SOURCE FILES INSPECTED

The following source files and dataset artifacts were inspected across the `backend/` and `data/` directories:

| Module / System Layer | Source File Path | Key Classes / Functions Inspected | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Legal Mapping Engine** | [`backend/app/services/legal_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/legal_mapping.py#L22-L174) | `LegalMappingService.get_legal_provisions()` | Maps predicted BERT incident categories to stored PostgreSQL `laws` records. |
| **Support Mapping Engine** | [`backend/app/services/support_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/support_mapping.py#L24-L215) | `SupportMappingService.get_support_services()` | Performs category and location-aware (District/State/National) filtering and deterministic ranking. |
| **Anti-Hallucination Gating** | [`backend/app/services/anti_hallucination.py`](file:///d:/Abhera(Mini)/backend/app/services/anti_hallucination.py#L24-L227) | `AntiHallucinationService.validate_and_gate_results()` | Verifies legal and support entries against PostgreSQL tables verbatim; gates out unverified entries. |
| **Report Generation Engine** | [`backend/app/services/report_generator.py`](file:///d:/Abhera(Mini)/backend/app/services/report_generator.py#L77-L415) | `ReportGeneratorService.generate_report()` | Assembles the 10 conceptual report sections into a structured JSON/PDF survivor report payload. |
| **Dataset Importer & Seeding** | [`backend/app/services/data_importer.py`](file:///d:/Abhera(Mini)/backend/app/services/data_importer.py#L132-L244) | `DatasetImporter.process_laws_csv()`, `process_support_services_csv()` | Validates CSV schemas, removes exact duplicates, and syncs `laws.csv` and `support_services.csv` into PostgreSQL. |
| **Laws Database Model** | [`backend/app/models/law.py`](file:///d:/Abhera(Mini)/backend/app/models/law.py#L6-L19) | SQLAlchemy `Law` model (`__tablename__ = "laws"`) | Maps database schema for legal provisions (`act_name`, `section_number`, `section_text`, `applicable_label`). |
| **Support Service Model** | [`backend/app/models/support_service.py`](file:///d:/Abhera(Mini)/backend/app/models/support_service.py#L6-L21) | SQLAlchemy `SupportService` model (`__tablename__ = "support_services"`) | Maps database schema for support resources (`service_type`, `name`, `contact_number`, `state`, `district`, `applicable_label`). |
| **Laws Dataset CSV** | [`data/laws.csv`](file:///d:/Abhera(Mini)/data/laws.csv#L1-L34) | 33 statutory records | Source ground-truth legal dataset. |
| **Support Services CSV** | [`data/support_services.csv`](file:///d:/Abhera(Mini)/data/support_services.csv#L1-L14) | 13 helpline / portal / shelter records | Source ground-truth support services dataset. |
| **BERT Classification Inference** | [`ml/inference/predict_bert.py`](file:///d:/Abhera(Mini)/ml/inference/predict_bert.py#L232-L360) | `BERTIncidentPredictor.predict()` | Generates predicted category codes (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`, `FPM`, `DOWRY`). |
| **NER Entity Inference** | [`ml/inference/predict_ner.py`](file:///d:/Abhera(Mini)/ml/inference/predict_ner.py#L127-L197) | `NEREntityPredictor.predict()` | Extracts structured entity locations (`LOCATION`, `STATE`, `DISTRICT`). |

---

# 3. OVERALL RECOMMENDATION ARCHITECTURE & DATA FLOW

The legal information and support service recommendation architecture operates as a **strictly non-generative, database-grounded retrieval pipeline**:

```
                       RAW SURVIVOR INCIDENT NARRATIVE
                                      │
                                      ▼
                      BERT MULTI-LABEL CLASSIFICATION
                     (ml/inference/predict_bert.py)
                                      │
                   Predicted Incident Categories L_pred
                [e.g., "DV" (Domestic Violence), "CA" (Cyber Abuse)]
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
LEGAL MAPPING ENGINE                                  SUPPORT MAPPING ENGINE
(backend/app/services/legal_mapping.py)              (backend/app/services/support_mapping.py)
- Input: L_pred = ["DV", "CA"]                        - Input: L_pred, Location (State/District)
- Normalization: LABEL_TO_APPLICABLE_CODE             - Category Matching: regex VALID_CATEGORY_PATTERNS
- Database Query: PostgreSQL laws table               - Location Scoring:
  Law.applicable_label.in_(["DV", "CA"])                * District Match  : 300 points
- Deduplication: (act_name, section, label)             * State Match     : 200 points
- Fallback: "Information not available..."              * National Fallback: 100 points
                                                        * Local Mismatch  : 0 points (Excluded)
                                                      - Sorting: (-location_score, rec.id)
           │                                                     │
           └──────────────────────────┬──────────────────────────┘
                                      ▼
                    ANTI-HALLUCINATION GATING LAYER
               (backend/app/services/anti_hallucination.py)
               - Verifies required non-empty fields
               - Queries PostgreSQL database tables for exact matches
               - Validates section_text and contact_number verbatim
               - Gates out unverified entries and records audit metrics
                                      │
                                      ▼
                     REPORT GENERATION SERVICE ENGINE
               (backend/app/services/report_generator.py)
               - Section 05: Relevant Legal Information (Database-backed)
               - Section 07: Available Support Services (Database-backed)
               - Generates non-generative JSON & PDF Survivor Report
```

---

# 4. BERT-TO-LEGAL MAPPING & RETRIEVAL PROCEDURE

### 4.1 Input and Label Normalization
`LegalMappingService.get_legal_provisions()` in `backend/app/services/legal_mapping.py` accepts a list of predicted BERT category strings (e.g., `["Domestic Violence", "CA"]`).

Input labels are normalized to the exact `applicable_label` codes stored in the database using `LABEL_TO_APPLICABLE_CODE`:

```python
LABEL_TO_APPLICABLE_CODE = {
    "DV": "DV", "DOMESTIC VIOLENCE": "DV",
    "SH": "SH", "SEXUAL HARASSMENT": "SH",
    "ST": "ST", "STALKING": "ST",
    "CA": "CA", "CYBER HARASSMENT": "CA", "CYBER ABUSE": "CA",
    "WH": "WH", "WORKPLACE HARASSMENT": "WH",
    "OV": "OV", "OTHER HARASSMENT": "OV", "OTHER SAFETY": "OV", "GENERAL": "OV",
    "FPM": "FPM", "FALSE PROMISE OF MARRIAGE": "FPM",
    "DOWRY": "DOWRY", "DOWRY HARASSMENT": "DOWRY"
}
```

### 4.2 Database Query Execution
The normalized codes are passed into an ORM query against the PostgreSQL `laws` table:

```python
# Parameterized ORM Query in legal_mapping.py (Lines 115-120)
raw_records = (
    db.query(Law)
    .filter(Law.applicable_label.in_(normalized_codes))
    .order_by(Law.id.asc())
    .all()
)
```

### 4.3 Multi-Label Aggregation & Deduplication
When BERT predicts multiple categories (e.g., `DV` + `CA`), the SQL `IN` operator retrieves statutory records matching **either** category. The retrieved records are deduplicated based on `(act_name, section_number, applicable_label)` while preserving the exact text stored in the database:

```python
dedup_key = (rec.act_name, rec.section_number, rec.applicable_label)
if dedup_key not in seen_keys:
    seen_keys.add(dedup_key)
    legal_info_list.append({
        "act": rec.act_name,
        "section": str(rec.section_number),
        "description": rec.section_text,
        "applicable_label": rec.applicable_label,
        "source": "database",
    })
```

### 4.4 Mandatory Legal Fallback
If no predicted labels match or no records are returned by the database, the service returns the exact mandatory fallback message:
$$\text{NO\_MATCH\_MESSAGE} = \text{"Information not available in the provided knowledge base."}$$

```json
{
  "matched": false,
  "legal_information": [],
  "message": "Information not available in the provided knowledge base."
}
```

---

# 5. LEGAL KNOWLEDGE BASE SPECIFICATIONS

The legal knowledge base is stored in `data/laws.csv` and synchronized into the PostgreSQL `laws` database table by `backend/app/services/data_importer.py`.

### 5.1 Dataset Schema (`data/laws.csv`)
* **`act_name` (String):** Statutory legislation title (e.g., `BNS`, `IPC`, `DV_ACT`, `IT_ACT`, `POSH_ACT`, `DOWRY_ACT`).
* **`section_number` (String):** Legal section number (e.g., `85`, `498A`, `3`, `66E`, `2n`).
* **`section_text` (Text):** Verbatim statutory text of the legal provision.
* **`applicable_label` (String):** Primary incident category code (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`, `DOWRY`, `FPM`).

### 5.2 Record Inventory
* **Total Statutory Records:** 33 records.
* **Act Distribution:**
  * **Bharatiya Nyaya Sanhita (BNS):** 9 sections (Sections 64, 74, 75, 77, 78, 79, 85, 86, 87).
  * **Indian Penal Code (IPC):** 6 sections (Sections 354, 354A, 354D, 376, 498A, 509).
  * **Protection of Women from Domestic Violence Act (DV_ACT):** 5 sections (Sections 3, 12, 18, 19, 20).
  * **Information Technology Act (IT_ACT):** 4 sections (Sections 66E, 67, 67A, 72).
  * **Sexual Harassment of Women at Workplace Act (POSH_ACT):** 5 sections (Sections 2n, 4, 9, 11, 13).
  * **Dowry Prohibition Act (DOWRY_ACT):** 2 sections (Sections 3, 4).
  * **Criminal Intimidation (BNS 351):** 1 section (Section 351).

---

# 6. SUPPORT SERVICE MAPPING & LOCATION SELECTION

`SupportMappingService.get_support_services()` in `backend/app/services/support_mapping.py` maps predicted incident categories and location entities (`state`, `district`) to verified support resources stored in PostgreSQL.

### 6.1 Category Matching Strategy
Category matching checks whether a record's `applicable_label` field satisfies regex patterns for any predicted label via `is_label_applicable()`:

```python
VALID_CATEGORY_PATTERNS = {
    "DV": r"\b(DV|DOMESTIC VIOLENCE|DOWRY|SHELTER|OSC|ALL EMERGENCIES)\b",
    "SH": r"\b(SH|SEXUAL HARASSMENT|SEXUAL ASSAULT|RAPE|MODESTY|ALL EMERGENCIES)\b",
    "ST": r"\b(ST|STALKING|OBSCENE|CYBERCRIME|ALL EMERGENCIES)\b",
    "CA": r"\b(CA|CYBER|CYBERCRIME|OBSCENE|POSH|ALL EMERGENCIES)\b",
    "WH": r"\b(WH|POSH|WORKPLACE|SEXUAL HARASSMENT|ALL EMERGENCIES)\b",
    "OV": r"\b(OV|OTHER|GENERAL|EMERGENCIES)\b",
    ...
}
```

### 6.2 Location-Aware Deterministic Scoring Logic
Support services are ranked using a **deterministic location-scoring function**:

$$S(\text{rec}, \text{state}, \text{district}) = \begin{cases} 
300 & \text{if } \text{rec.district} = \text{district} \quad (\text{Exact District Match}) \\
200 & \text{if } \text{rec.state} = \text{state} \quad (\text{State Match}) \\
100 & \text{if } \text{rec.state/district} \in \{\text{"National"}, \text{"All India"}\} \quad (\text{National Fallback}) \\
0 & \text{otherwise} \quad (\text{Local Mismatch — Excluded})
\end{cases}$$

### 6.3 Ranking & Sorting Protocol
1. Candidates with $S > 0$ are collected into tuples: `(location_score, rec.id, rec)`.
2. Candidates are sorted in descending order of location score, breaking ties by database ID ascending:
   ```python
   candidates.sort(key=lambda x: (-x[0], x[1]))
   ```
3. If no user location (`state`/`district`) is provided, national helplines ($S = 100$) are returned automatically as the default fallback.
4. If no records match, the service returns the exact mandatory fallback message:
   $$\text{NO\_MATCH\_MESSAGE} = \text{"No matching support service was found in the available support-services database."}$$

---

# 7. SUPPORT SERVICE KNOWLEDGE BASE SPECIFICATIONS

The support services knowledge base is stored in `data/support_services.csv` and synchronized into the PostgreSQL `support_services` table.

### 7.1 Dataset Schema (`data/support_services.csv`)
* **`service_type` (String):** Category of support (`Helpline`, `Portal`, `Shelter`, `OSC`).
* **`name` (String):** Official title of organization or helpline.
* **`contact_number` (String):** Phone number (e.g., `181`, `112`, `1930`, `14490`) or web portal URL.
* **`state` (String):** State jurisdiction (`National`, `Delhi`, etc.).
* **`district` (String):** District jurisdiction (`All India`, `New Delhi`, etc.).
* **`applicable_label` (String):** Associated incident category keywords.

### 7.2 Record Inventory
* **Total Support Service Records:** 13 records.
* **Service Breakdown:**
  * **National 24x7 Women Helpline (NCW):** `14490` (National / All India)
  * **Women Helpline (Sambal - Mission Shakti):** `181` (National / All India)
  * **Emergency Response Support System (ERSS):** `112` (National / All India)
  * **Anti-Obscene Calls / Women Helpline:** `1091` (National / All India)
  * **National Cyber Crime Helpline:** `1930` (National / All India)
  * **National Cyber Crime Reporting Portal:** `https://www.cybercrime.gov.in` (Portal)
  * **SHe-Box (Sexual Harassment electronic Box):** `https://shebox.wcd.gov.in` (Portal)
  * **Childline India:** `1098` (National / All India)
  * **National Human Rights Commission (NHRC):** `14433` (National / All India)
  * **Mission Shakti Portal:** `https://missionshakti.wcd.gov.in` (Portal)
  * **Shakti Shalini NGO Shelter & Helpline:** `10920 / 011-24373737` (Delhi / New Delhi)
  * **Delhi Commission for Women (DCW) Helpline:** `011-23379181` (Delhi / New Delhi)

---

# 8. ANTI-HALLUCINATION GATING MECHANISM

The `AntiHallucinationService` in `backend/app/services/anti_hallucination.py` acts as a strict verification layer between raw model predictions/queries and final report assembly.

### 8.1 Verification Pipeline

```
       RAW RETRIEVED LEGAL / SUPPORT PAYLOAD
                         │
                         ▼
        1. STRUCTURE & FIELD VALIDATION
           - Must be a valid dictionary.
           - Required fields must NOT be null or empty string.
                         │
                         ▼
        2. POSTGRESQL TABLE VERBATIM MATCHING
           - Query PostgreSQL laws / support_services table for exact match.
           - Legal Check: Match Law.act_name AND Law.section_number.
           - Support Check: Match SupportService.name AND SupportService.contact_number.
                         │
                         ▼
        3. VERBATIM CONTENT VERIFICATION
           - Legal: Verify item["description"] == db_match.section_text.strip().
           - Reject modified, summarized, or AI-rewritten legal text.
                         │
                         ▼
        4. GATING & AUDIT ATTACHMENT
           - Validated records -> Accepted list.
           - Unverified records -> Rejected list + log rejection reason.
           - Attach audit metadata: {received, accepted, rejected, rejection_reasons}.
```

### 8.2 Architectural Distinction
* **Machine Learning Models (BERT / NER):** Responsible **ONLY** for sequence classification (predicting category codes `DV`, `SH`) and entity extraction (extracting `LOCATION` strings).
* **Database & Anti-Hallucination Layer:** Responsible **EXCLUSIVELY** for providing, verifying, and gating statutory legal texts and support service contact details. **No generative model ever writes legal section text or invents phone numbers.**

---

# 9. QUESTIONNAIRE ENGINE & REPORT GENERATOR INTERACTION

### 9.1 Question Engine Complementarity
In `backend/app/services/report_generator.py`, questionnaire answers collected during the interactive chat complement NER extractions:
* If NER extractions miss location details, `_extract_location_or_platform_from_answers()` extracts `LOCATION` or `PLATFORM` entries from user questionnaire responses.
* Extracted location details populate `state` and `district` parameters for support service mapping.

### 9.2 Report Generation Assembly
`ReportGeneratorService.generate_report()` compiles 10 structured sections into a finalized survivor report:
1. **Incident Summary:** Raw narrative text.
2. **Identified Incident Type:** Predicted BERT labels list.
3. **Extracted Information:** NER entities (`PERP_REL`, `LOCATION`, `TIME_FREQ`, `PLATFORM`, `EVIDENCE`, `LAW_SEC`).
4. **User-Provided Details:** Formatted questionnaire Q&A pairs.
5. **Relevant Legal Information:** Anti-hallucination validated PostgreSQL legal provisions (Section 05).
6. **Suggested Next Steps:** Standardized safety preservation guidelines.
7. **Available Support Services:** Anti-hallucination validated PostgreSQL support helplines (Section 07).
8. **Evidence / Information Notes:** Aggregated evidence notes and NER `EVIDENCE` spans.
9. **Model Prediction Information:** Metadata detailing model names (`bert-base-uncased`), tasks, and database provenance.
10. **Disclaimer:** Official legal disclaimer.

---

# 10. FORMAL ALGORITHMS

```markdown
Algorithm 1: Knowledge-Base Legal Mapping

Input : Predicted incident category codes L_pred
        PostgreSQL laws database table K_law
Output: Validated legal provisions list R_legal or fallback message

1.  if L_pred is null or empty then
        return { matched: false, legal_information: [], message: NO_MATCH_MESSAGE }
    end if

2.  Normalize input category codes:
    normalized_codes <- []
    for each label in L_pred do
        code <- LABEL_TO_APPLICABLE_CODE.get(uppercase(label))
        if code is not null and code not in normalized_codes then
            normalized_codes.append(code)
        end if
    end for

3.  if normalized_codes is empty then
        return { matched: false, legal_information: [], message: NO_MATCH_MESSAGE }
    end if

4.  Execute parameterized ORM query against PostgreSQL:
    raw_records <- K_law.query().filter(Law.applicable_label IN normalized_codes).order_by(Law.id ASC)

5.  if raw_records is empty then
        return { matched: false, legal_information: [], message: NO_MATCH_MESSAGE }
    end if

6.  Format and deduplicate records:
    seen_keys <- empty_set()
    R_legal <- empty_list()
    for each rec in raw_records do
        key <- (rec.act_name, rec.section_number, rec.applicable_label)
        if key not in seen_keys then
            seen_keys.add(key)
            R_legal.append({
                act: rec.act_name,
                section: rec.section_number,
                description: rec.section_text,
                applicable_label: rec.applicable_label,
                source: "database"
            })
        end if
    end for

7.  return { matched: true, legal_information: R_legal }
```

```markdown
Algorithm 2: Location- and Category-Aware Support Service Mapping

Input : Predicted incident category codes L_pred
        User location (state, district)
        PostgreSQL support_services database table K_support
Output: Ranked, validated support services list R_support or fallback message

1.  if L_pred is null or empty then
        return { matched: false, support_services: [], message: NO_MATCH_MESSAGE }
    end if

2.  Fetch all support service records from PostgreSQL:
    all_records <- K_support.query().all()

3.  Filter and score candidates based on category and location:
    candidates <- empty_list()
    for each rec in all_records do
        if is_label_applicable(L_pred, rec.applicable_label) then
            score <- 0
            if district is not null and rec.district == district then
                score <- 300   // District Match
            else if state is not null and rec.state == state then
                score <- 200   // State Match
            else if rec.state in ["National", "All India"] then
                score <- 100   // National Fallback
            end if

            if score > 0 then
                candidates.append((score, rec.id, rec))
            end if
        end if
    end for

4.  if candidates is empty then
        return { matched: false, support_services: [], message: NO_MATCH_MESSAGE }
    end if

5.  Sort candidates deterministically by score descending, then ID ascending:
    candidates.sort_by(-score, id)

6.  Deduplicate and format output list R_support:
    seen_keys <- empty_set()
    R_support <- empty_list()
    for each (score, id, rec) in candidates do
        key <- (rec.name, rec.service_type, rec.contact_number, rec.state, rec.district)
        if key not in seen_keys then
            seen_keys.add(key)
            R_support.append({
                service_name: rec.name,
                service_type: rec.service_type,
                contact_number: rec.contact_number,
                state: rec.state,
                district: rec.district,
                applicable_label: rec.applicable_label,
                source: "database"
            })
        end if
    end for

7.  return { matched: true, support_services: R_support }
```

---

# 11. MATHEMATICAL / RULE-BASED FORMULATION

### Legal Provision Retrieval Function:
Let $\hat{L} \subseteq \{\text{DV, SH, ST, CA, WH, OV, FPM, DOWRY}\}$ be the set of predicted incident categories. The set of retrieved legal provisions $R_{\text{legal}}$ is given by:

$$R_{\text{legal}} = \left\{ r \in \mathcal{K}_{\text{law}} \;\middle|\; \text{norm}(r.\text{applicable\_label}) \in \hat{L} \right\}$$

Deduplicated over tuples $(r.\text{act\_name}, r.\text{section\_number}, r.\text{applicable\_label})$.

### Support Service Scoring & Selection Function:
Let $s_{\text{user}}$ and $d_{\text{user}}$ denote the user's normalized state and district, respectively. For each service record $r \in \mathcal{K}_{\text{support}}$ satisfying category applicability $\text{App}(r, \hat{L}) = \text{True}$:

$$\text{Score}(r, s_{\text{user}}, d_{\text{user}}) = \begin{cases}
300 & \text{if } r.\text{district} = d_{\text{user}} \\
200 & \text{if } r.\text{state} = s_{\text{user}} \\
100 & \text{if } r.\text{state} \in \{\text{"National"}, \text{"All India"}\} \\
0 & \text{otherwise}
\end{cases}$$

The ranked result set $R_{\text{support}}$ is defined as:

$$R_{\text{support}} = \text{Deduplicate}\left( \text{SortBy}\left( \{ r \in \mathcal{K}_{\text{support}} \mid \text{Score}(r) > 0 \}, \, (-\text{Score}, \text{id}) \right) \right)$$

---

# 12. PAPER-READY RESEARCH SECTION

*The text below is formatted in formal academic prose for direct inclusion in the research paper:*

```markdown
### 3.5 Legal Knowledge Retrieval and Support Service Mapping

To eliminate generative hallucination risk in high-stakes legal domain applications, ABHERA separates pattern recognition from information retrieval. Machine learning models (fine-tuned BERT and NER token classifiers) are restricted to sequence classification and entity extraction tasks. Statutory legal provisions and support service recommendations are retrieved exclusively from an empirically verified PostgreSQL knowledge base.

Legal provision mapping is executed by `LegalMappingService`. Predicted multi-label category codes $\hat{L} \subseteq \{\text{DV, SH, ST, CA, WH, OV}\}$ are normalized and evaluated via parameterized SQL queries against the `laws` table (containing 33 verified statutory sections spanning the Bharatiya Nyaya Sanhita, Indian Penal Code, DV Act, IT Act, POSH Act, and Dowry Prohibition Act). Matching statutory provisions are retrieved, deduplicated by act title and section number, and presented with verbatim legal text. If no predicted categories match stored records, the system returns a mandatory fallback message ("Information not available in the provided knowledge base"), preventing ungrounded legal generation.

Support service mapping is conducted by `SupportMappingService`. Candidate support resources (13 verified national helplines, emergency services, and regional shelters) are matched based on incident applicability and ranked using a deterministic location-scoring algorithm: exact district matches receive top priority (Score: 300), state-level matches receive secondary priority (Score: 200), and national 24x7 helplines (e.g., 181, 112, 1930) serve as universal fallbacks (Score: 100).

All retrieved payloads pass through an automated `AntiHallucinationService` gating layer prior to report synthesis. This layer queries the database to verify that every retrieved act title, section number, section text, service name, and contact number matches an empirical database record verbatim. Any fabricated or altered entry is gated out, and provenance audit metrics are logged. The validated information is assembled into a structured 10-section survivor incident report.
```

---

# 13. VIVA-READY EXAMINATION QUESTIONS & ANSWERS

### Q1: How are laws suggested in ABHERA?
**Answer:** Laws are retrieved through category-based database lookup in `LegalMappingService`. Predicted BERT categories (`DV`, `SH`, etc.) are normalized and matched against stored statutory records in the PostgreSQL `laws` table using parameterized ORM queries.

### Q2: Does BERT generate legal sections or text?
**Answer:** **No, BERT does NOT generate any legal text.** BERT is used strictly for incident category classification. All legal titles, section numbers, and section texts are retrieved verbatim from the verified database.

### Q3: Where do legal provisions come from?
**Answer:** Legal provisions are stored in `data/laws.csv` (33 records) and synchronized into the PostgreSQL `laws` database table by `data_importer.py`.

### Q4: What happens when BERT predicts multiple categories (e.g., DV + CA)?
**Answer:** The ORM query evaluates `Law.applicable_label.in_(["DV", "CA"])`, retrieving legal provisions matching **either** category. Results are combined, deduplicated by act and section number, and presented together.

### Q5: How are support services selected?
**Answer:** Support services are selected using category matching and a **deterministic location-scoring algorithm**: District match (300 pts), State match (200 pts), National 24x7 helpline fallback (100 pts).

### Q6: Does location affect support service selection?
**Answer:** Yes. Extracted location entities (State/District) boost local regional helplines and shelters above national services. If no location is provided, national helplines (e.g., 181, 112, 1930) are returned as the default fallback.

### Q7: Where are support service contact details stored?
**Answer:** Support service details are stored in `data/support_services.csv` (13 records) and synchronized into the PostgreSQL `support_services` table.

### Q8: How does ABHERA prevent hallucinated laws or fake phone numbers?
**Answer:** Through `AntiHallucinationService`. Before inclusion in reports, every legal section and support contact is queried against PostgreSQL. Any record whose title, section, text, or contact number does not match a database row verbatim is rejected.

### Q9: Can the system invent a new helpline phone number?
**Answer:** **No.** Phone numbers and URLs are hardcoded in the verified database and validated verbatim by the anti-hallucination gating layer.

### Q10: What happens when no matching law exists for a query?
**Answer:** The system returns the exact mandatory fallback message: `"Information not available in the provided knowledge base."`

### Q11: What happens when no matching support service exists?
**Answer:** The system returns the exact mandatory fallback message: `"No matching support service was found in the available support-services database."`

### Q12: Is an LLM used to write legal recommendations?
**Answer:** **No.** Large language models are not used to write or summarize statutory legal provisions.

### Q13: What is the difference between BERT classification and legal mapping?
**Answer:** BERT classification predicts **what category of incident** occurred (e.g., `DV`). Legal mapping retrieves the **exact statutory legal sections** (e.g., BNS 85, IPC 498A) corresponding to that category from the database.

### Q14: How does the final report obtain legal and support information?
**Answer:** `ReportGeneratorService.generate_report()` invokes `validate_payload()`, which executes database mapping, runs anti-hallucination validation, and populates Section 05 (Legal Information) and Section 07 (Support Services).

---

# 14. SOURCE TRACEABILITY MATRIX

| Feature / Module | Source File Path | Function / Class | Implementation Behavior | Research Paper Implication |
| :--- | :--- | :--- | :--- | :--- |
| **Legal Mapping** | [`backend/app/services/legal_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/legal_mapping.py#L66-L149) | `LegalMappingService.get_legal_provisions()` | Queries `Law` model via `.filter(Law.applicable_label.in_(normalized_codes))`; returns fallback `NO_MATCH_MESSAGE` if empty. | Category-based database lookup of statutory provisions. |
| **Support Mapping** | [`backend/app/services/support_mapping.py`](file:///d:/Abhera(Mini)/backend/app/services/support_mapping.py#L75-L188) | `SupportMappingService.get_support_services()` | Category matching via regex + location scoring (District=300, State=200, National=100); sorts by `(-score, id)`. | Category- and location-aware deterministic support selection. |
| **Anti-Hallucination** | [`backend/app/services/anti_hallucination.py`](file:///d:/Abhera(Mini)/backend/app/services/anti_hallucination.py#L116-L199) | `AntiHallucinationService.validate_and_gate_results()` | Queries PostgreSQL `Law` and `SupportService` tables for verbatim matching; gates out unverified entries. | Automated anti-hallucination validation layer. |
| **Report Generation** | [`backend/app/services/report_generator.py`](file:///d:/Abhera(Mini)/backend/app/services/report_generator.py#L288-L350) | `ReportGeneratorService.generate_report()` | Integrates validated legal & support payloads into Sections 05 and 07 of the 10-section report structure. | Non-generative structured report assembly. |
| **Laws Dataset** | [`data/laws.csv`](file:///d:/Abhera(Mini)/data/laws.csv#L1-L34) | 33 rows | Defines `act_name`, `section_number`, `section_text`, `applicable_label`. | Source ground-truth statutory legal dataset. |
| **Support Dataset** | [`data/support_services.csv`](file:///d:/Abhera(Mini)/data/support_services.csv#L1-L14) | 13 rows | Defines `service_type`, `name`, `contact_number`, `state`, `district`, `applicable_label`. | Source ground-truth support service dataset. |

---

ACTUAL LEGAL AND SUPPORT RECOMMENDATION STATUS: VERIFIED
