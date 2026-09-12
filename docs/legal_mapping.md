# Legal Mapping Engine Technical Specification & Architecture

## 1. Purpose

The **Legal Mapping Engine** provides a strict, database-grounded mechanism for mapping predicted incident categories to official legal provisions stored in the PostgreSQL database.

> **Core Principle**:  
> *"The model predicts incident categories; the database provides the legal information."*

The engine guarantees that the AI assistant **never generates, infers, or hallucinates legal sections, Acts, penalties, or advice**. All legal provisions returned by the engine are empirical database records loaded directly from the PostgreSQL `laws` table.

---

## 2. Input Specification

The Legal Mapping Engine accepts predicted BERT incident categories and optional incident metadata:

```json
{
    "predicted_labels": [
        "Domestic Violence",
        "Workplace Harassment"
    ],
    "incident_information": {
        "PERP_REL": "husband",
        "LOCATION": "home"
    },
    "submission_id": "SUB-849201"
}
```

Input label representations supported:
- **Short Codes**: `"DV"`, `"SH"`, `"ST"`, `"CA"`, `"WH"`, `"OV"`
- **Canonical Names**: `"Domestic Violence"`, `"Sexual Harassment"`, `"Stalking"`, `"Cyber Harassment"`, `"Workplace Harassment"`, `"Other Harassment"` / `"Other Safety"`

---

## 3. Database Source & Schema

The service queries the PostgreSQL `laws` table (populated from [`data/laws.csv`](file:///D:/Abhera(Mini)/data/laws.csv)) via SQLAlchemy ORM in [`backend/app/models/law.py`](file:///D:/Abhera(Mini)/backend/app/models/law.py):

### Table: `laws`
| Column Name | Database Type | Index | Description |
| :--- | :--- | :---: | :--- |
| `id` | `INTEGER` | PK | Auto-incrementing Primary Key |
| `act_name` | `VARCHAR(100)` | Yes | Statute/Act Name (e.g., `BNS`, `IPC`, `DV_ACT`, `POSH_ACT`, `IT_ACT`, `DOWRY_ACT`) |
| `section_number` | `VARCHAR(50)` | No | Section Number (e.g., `85`, `498A`, `354`, `66E`, `2n`, `4`) |
| `section_text` | `TEXT` | No | Exact legal section description from database |
| `applicable_label` | `VARCHAR(50)` | Yes | Category code (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`) |
| `created_at` | `TIMESTAMP` | No | Record creation timestamp |

---

## 4. Matching Logic

Matching operates deterministically using parameterized SQLAlchemy ORM queries:

```
BERT Predicted Labels (e.g. ["Domestic Violence", "Stalking"])
                          ↓
        Normalize to Applicable Codes (["DV", "ST"])
                          ↓
    SQL: SELECT * FROM laws WHERE applicable_label IN ('DV', 'ST') ORDER BY id ASC
                          ↓
     Deduplicate Records & Preserve Exact Database Field Values
                          ↓
             Format Structured Output Payload
```

---

## 5. Multiple-Label Handling

When BERT predicts multiple incident categories (e.g., `["Domestic Violence", "Workplace Harassment"]`):
1. Input labels are normalized to codes `["DV", "WH"]`.
2. The service queries all matching rows for both categories in a single parameterized SQL query.
3. Records are returned grouped or sorted deterministically.
4. Duplicate legal provisions (if the same record matches multiple input labels) are automatically removed.

---

## 6. No-Match & Fallback Behavior

If no legal records match the input labels, or if an unknown/unmapped category is provided (e.g., `["UNKNOWN_CATEGORY"]`):
1. `matched` is set to `false`.
2. `legal_information` is set to an empty list `[]`.
3. `message` returns **EXACTLY**:
   ```
   Information not available in the provided knowledge base.
   ```

---

## 7. Anti-Hallucination Rules

The Legal Mapping Engine strictly enforces 8 mandatory anti-hallucination rules:

1. **Zero Generative Code**: No LLM or generative logic is used to output legal provisions.
2. **Zero Hardcoded Assumed Mappings**: No hardcoded dictionary maps `DV -> Section XYZ`. All provisions come from SQL queries on `laws`.
3. **No External APIs or Web Search**: The engine operates 100% locally against the PostgreSQL database.
4. **Preservation of Database Values**: Values for `act`, `section`, and `description` are taken verbatim from database columns (`act_name`, `section_number`, `section_text`) without rewriting or editing.
5. **Exact Fallback String**: Unmapped or zero-result queries return the exact string `"Information not available in the provided knowledge base."`.
6. **No Silent Fallback Guessing**: The engine never attempts to guess a default or fallback legal section.
7. **Input Normalization Only**: Label normalization only maps string representations to standard category codes (`"Domestic Violence"` -> `"DV"`).
8. **SQL Injection Safety**: Uses SQLAlchemy ORM parameterized queries (`filter(Law.applicable_label.in_(...))`).

---

## 8. Return Structure

### Successful Match Payload
```json
{
    "matched": true,
    "legal_information": [
        {
            "act": "BNS",
            "section": "85",
            "description": "Whoever, being the husband or relative of the husband of a woman, subjects such woman to cruelty...",
            "applicable_label": "DV",
            "source": "database"
        },
        {
            "act": "IPC",
            "section": "498A",
            "description": "Whoever, being the husband or the relative of the husband of a woman...",
            "applicable_label": "DV",
            "source": "database"
        }
    ]
}
```

### No Match / Unknown Category Payload
```json
{
    "matched": false,
    "legal_information": [],
    "message": "Information not available in the provided knowledge base."
}
```

---

## 9. Test Coverage Summary

All 9 unit and integration test cases in [`backend/tests/test_legal_mapping.py`](file:///D:/Abhera(Mini)/backend/tests/test_legal_mapping.py) passed:

1. `test_01_matching_provision`: Verifies valid category returns database records.
2. `test_02_multiple_matching_provisions`: Verifies category with 10 matching provisions (`DV`) returns all 10 rows.
3. `test_03_multiple_predicted_labels`: Verifies multi-label query (`DV` + `WH`) returns records for both categories.
4. `test_04_no_matching_provision`: Verifies unknown category returns exact fallback message.
5. `test_05_invalid_label`: Verifies `None` or invalid types return safe fallback without crashing.
6. `test_06_empty_labels`: Verifies empty list `[]` returns safe fallback.
7. `test_07_duplicate_labels`: Verifies duplicate input labels do not produce duplicate output records.
8. `test_08_database_values_are_preserved`: Verifies returned Act, Section, and Description fields match database columns verbatim.
9. `test_09_anti_hallucination`: Explicitly verifies `["UNKNOWN_CATEGORY"]` returns zero guessed laws and exact fallback message.

---

## 10. Example Flow Walkthroughs

### Example 1: Domestic Violence (`DV`)
- **Input**: `["Domestic Violence"]`
- **Normalized Code**: `["DV"]`
- **Database Output**: 10 legal records (`BNS 85`, `IPC 498A`, `BNS 86`, `DV_ACT 3`, `DV_ACT 12`, `DV_ACT 18`, `DV_ACT 19`, `DV_ACT 20`, `DOWRY_ACT 3`, `DOWRY_ACT 4`).

### Example 2: Workplace Harassment (`WH`)
- **Input**: `["Workplace Harassment"]`
- **Normalized Code**: `["WH"]`
- **Database Output**: 5 legal records (`POSH_ACT 2n`, `POSH_ACT 4`, `POSH_ACT 9`, `POSH_ACT 11`, `POSH_ACT 13`).

### Example 3: Unmapped / Unknown Category
- **Input**: `["Traffic Violation"]`
- **Normalized Code**: `[]`
- **Output**: `{"matched": false, "legal_information": [], "message": "Information not available in the provided knowledge base."}`
