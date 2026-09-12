# Support Service Mapping Engine Technical Specification & Architecture

## 1. Purpose

The **Support Service Mapping Engine** provides a location-aware, database-grounded mechanism for retrieving verified emergency helplines, shelters, One Stop Centers (OSC), and reporting portals based on predicted incident categories and user location (State / District).

> **Core System Architecture**:  
> *"The support mapping service returns only verified records present in the provided support-services knowledge base."*

The engine guarantees that the AI assistant **never invents support organizations, phone numbers, email addresses, websites, helplines, or addresses**. All support records returned by the engine are empirical database rows loaded directly from the PostgreSQL `support_services` table.

---

## 2. Dataset Source & Schema

The service queries the PostgreSQL `support_services` table (populated from [`data/support_services.csv`](file:///D:/Abhera(Mini)/data/support_services.csv)) via SQLAlchemy ORM in [`backend/app/models/support_service.py`](file:///D:/Abhera(Mini)/backend/app/models/support_service.py):

### Table: `support_services`
| Column Name | Database Type | Index | Description |
| :--- | :--- | :---: | :--- |
| `id` | `INTEGER` | PK | Auto-incrementing Primary Key |
| `service_type` | `VARCHAR(50)` | Yes | Type of service (`Helpline`, `Portal`, `Shelter`, `OSC`) |
| `name` | `VARCHAR(255)` | No | Verified Organization / Service Name |
| `contact_number` | `VARCHAR(255)` | No | Phone helpline number, URL, or contact details |
| `state` | `VARCHAR(100)` | Yes | Jurisdiction State (`National`, `Delhi`, etc.) |
| `district` | `VARCHAR(100)` | Yes | Jurisdiction District (`All India`, `New Delhi`, etc.) |
| `applicable_label` | `VARCHAR(255)` | Yes | Applicable category tags (`DV`, `SH`, `ST`, `CA`, `WH`, `OV`, etc.) |
| `created_at` | `TIMESTAMP` | No | Record creation timestamp |

---

## 3. Matching & Location Ranking Logic

Matching and ranking proceed deterministically:

```
BERT Predicted Labels + Location (State / District)
                         ↓
  Query All PostgreSQL support_services Database Rows
                         ↓
 Filter & Match Applicable Categories (is_label_applicable)
                         ↓
            Calculate Location Relevance Score:
  - Exact District Match (rec.district == target_district) → Score = 300
  - State Match (rec.state == target_state)                → Score = 200
  - National Fallback (rec.state in ["National", "All India"]) → Score = 100
  - Non-matching Local Service                             → Score = 0 (Excluded)
                         ↓
  Sort Candidates by Location Score (descending) & ID (ascending)
                         ↓
  Deduplicate & Return Structured Verified Support Service Objects
```

---

## 4. Multi-Label & Location Handling

### Category Matching
- Supports single or multiple BERT predicted categories (e.g. `["DV", "CA"]` or `["Domestic Violence", "Cyber Harassment"]`).
- Records matching any of the predicted categories are retrieved.
- Unmapped or unknown categories (e.g. `["UNKNOWN_CATEGORY"]`) do not match any database row.

### Location Handling
- **District Match**: Exact local district services (e.g., Shakti Shalini NGO Shelter in `New Delhi`, `Delhi`) are ranked highest (Score 300).
- **State Match**: State-level services are ranked second (Score 200).
- **National Fallback**: 24x7 National helplines (NCW 14490, Women Helpline 181, ERSS 112, Cyber Crime Helpline 1930) are available across all Indian states/districts (Score 100).
- **Location Integrity Protection**: Local services belonging to *other* states (e.g., a Delhi shelter when the user is in Telangana) are strictly excluded (`Score = 0`) to prevent misrepresenting non-local services as local.

---

## 5. No-Match & Fallback Behavior

If no support service matches the predicted categories or location, or if an unknown category is passed:
1. `matched` is set to `false`.
2. `support_services` is set to an empty list `[]`.
3. `message` returns **EXACTLY**:
   ```
   No matching support service was found in the available support-services database.
   ```

---

## 6. Anti-Hallucination Rules

The Support Service Mapping Engine strictly enforces 8 anti-hallucination rules:

1. **Zero Generative Code**: No LLM or generative text logic is used to construct support services.
2. **Zero Fabricated Organizations or Helplines**: Never invents phone numbers, helplines, websites, addresses, or emails.
3. **No External APIs or Web Search**: Operates 100% locally against PostgreSQL `support_services`.
4. **Preservation of Database Values**: Values for `service_name`, `service_type`, `contact_number`, `state`, and `district` come verbatim from database columns (`name`, `service_type`, `contact_number`, `state`, `district`).
5. **Exact Fallback String**: Zero-result queries return the exact string `"No matching support service was found in the available support-services database."`.
6. **No Location Hallucination**: Never claims a local organization exists in a district unless explicitly recorded in PostgreSQL.
7. **SQL Injection Safety**: Uses SQLAlchemy ORM parameterized queries.

---

## 7. Return Structure

### Successful Match Payload
```json
{
    "matched": true,
    "support_services": [
        {
            "service_name": "Shakti Shalini (NGO Shelter & Helpline)",
            "service_type": "Shelter",
            "contact_number": "10920 / 011-24373737",
            "state": "Delhi",
            "district": "New Delhi",
            "applicable_label": "Domestic Violence, Shelter, Counselling, Survivor Support",
            "source": "database"
        },
        {
            "service_name": "National Commission for Women (NCW) 24x7 Helpline",
            "service_type": "Helpline",
            "contact_number": "14490",
            "state": "National",
            "district": "All India",
            "applicable_label": "DV, SH, OV",
            "source": "database"
        }
    ]
}
```

### No Match Payload
```json
{
    "matched": false,
    "support_services": [],
    "message": "No matching support service was found in the available support-services database."
}
```

---

## 8. Test Coverage Summary

All 12 unit and integration test cases in [`backend/tests/test_support_mapping.py`](file:///D:/Abhera(Mini)/backend/tests/test_support_mapping.py) passed:

1. `test_01_category_match`: Verifies known category retrieves valid DB record.
2. `test_02_state_match`: Verifies state filtering (`Delhi`) includes state-specific services.
3. `test_03_district_match`: Verifies district filtering (`New Delhi`) ranks local district services.
4. `test_04_category_state_district_match`: Verifies highest-priority ranking for exact district match.
5. `test_05_multiple_predicted_labels`: Verifies multi-label queries (`DV` + `CA`) return matching records.
6. `test_06_no_match`: Verifies unknown category returns exact fallback message.
7. `test_07_missing_location`: Verifies missing location parameters fallback safely to National helplines.
8. `test_08_empty_labels`: Verifies empty list `[]` returns safe fallback.
9. `test_09_duplicate_labels`: Verifies duplicate input labels do not duplicate output records.
10. `test_10_database_values_preserved`: Verifies returned name, type, contact, and location fields match database columns verbatim.
11. `test_11_no_fabricated_contact_information`: Verifies no phone/helpline/contact fields are generated when no DB match exists.
12. `test_12_anti_hallucination`: Explicitly verifies `["UNKNOWN_CATEGORY"]`, `UnknownState`, `UnknownDistrict` returns 0 guessed organizations and exact fallback message.

---

## 9. Sample Flow Walkthroughs

### Example 1: Domestic Violence (`DV`) in New Delhi, Delhi
- **Input**: `predicted_labels = ["Domestic Violence"]`, `state = "Delhi"`, `district = "New Delhi"`
- **Matching Output**: 6 support services
  1. Shakti Shalini (Shelter, Delhi, New Delhi) — *Rank 1 (Exact District Match)*
  2. Delhi Commission for Women Helpline (OSC, Delhi, New Delhi) — *Rank 2 (Exact District Match)*
  3. NCW 24x7 Helpline (Helpline, National, All India) — *Rank 3 (National Fallback)*
  4. Women Helpline 181 (Helpline, National, All India) — *Rank 4 (National Fallback)*
  5. ERSS 112 (Helpline, National, All India) — *Rank 5 (National Fallback)*
  6. Mission Shakti Portal (Portal, National, All India) — *Rank 6 (National Fallback)*

### Example 2: Stalking (`ST`) in Hyderabad, Telangana
- **Input**: `predicted_labels = ["Stalking"]`, `state = "Telangana"`, `district = "Hyderabad"`
- **Matching Output**: 4 National helplines/portals available in Telangana (Delhi-only services excluded).

### Example 3: Unmapped / Unknown Category
- **Input**: `predicted_labels = ["UNKNOWN_CATEGORY"]`, `state = "UnknownState"`, `district = "UnknownDistrict"`
- **Output**: `{"matched": false, "support_services": [], "message": "No matching support service was found in the available support-services database."}`
