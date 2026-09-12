# Anti-Hallucination Safety Layer Technical Specification & Architecture

## 1. Architectural Core Principle

> **Fundamental Principle**:  
> *"The model predicts and extracts; the database verifies and provides the final legal and support information."*

The **Anti-Hallucination Layer** acts as a strict security, gating, and provenance verification firewall between AI model predictions (BERT / NER) and the final legal and support outputs. It guarantees that **no fake legal provisions, Acts, sections, helplines, organizations, phone numbers, or addresses** are ever presented to the user.

---

## 2. System Architecture & Flow

```
                      USER NARRATIVE
                            │
                            ▼
               BERT Multi-Label Classifier
                            │ (Predicted Categories: ["DV", "WH"])
                            ▼
              NER Entity Extractor (Extracted Entities)
                            │
                            ▼
                 Database Provenance Layer
               (SQLAlchemy ORM Parameterized Queries)
                 ┌──────────┴──────────┐
                 ▼                     ▼
          PostgreSQL `laws`   PostgreSQL `support_services`
                 └──────────┬──────────┘
                            ▼
            Anti-Hallucination Validation Layer
      (validate_legal_record & validate_support_record)
                 ┌──────────┴──────────┐
                 ▼                     ▼
        [100% Validated Items]   [Audit Metrics Log]
                 │
                 ▼
         FINAL REPORT / RESPONSE PAYLOAD
```

### System Anti-Pattern Prevention (Forbidden Architecture):
$$\text{USER} \xrightarrow{\quad\text{FORBIDDEN}\quad} \text{LLM Generative Model} \xrightarrow{\quad\text{FORBIDDEN}\quad} \text{Hallucinated Legal/Support Advice}$$

---

## 3. Threat Model & Mitigations

| Threat / Vulnerability | Vector | Anti-Hallucination Mitigation |
| :--- | :--- | :--- |
| **Generative Legal Section Invention** | LLM or model generating unverified Acts or Sections (e.g. `Section 999`) | **Provenance DB Check**: Every legal item must exist verbatim in PostgreSQL `laws` table (`act_name`, `section_number`). Unmatched items are **REJECTED**. |
| **Fake Helpline / Phone Number Generation** | Model hallucinating fake 10-digit helplines or fake NGO names | **DB Verification**: Every support service must exist verbatim in PostgreSQL `support_services` table (`name`, `contact_number`). Unmatched items are **REJECTED**. |
| **Location Misrepresentation** | Claiming a local Delhi shelter exists in Telangana | **Location Ranking & Filtering**: Local services belonging to non-matching states are excluded (`Score = 0`). Only matching local or National helplines are validated. |
| **Description Alteration** | Model rephrasing or modifying exact statutory text | **Verbatim Text Hash Matching**: `description` field must match `section_text` in database verbatim. Modified texts are **REJECTED**. |
| **Unknown Category Guessing** | Model outputting unmapped categories (e.g. `["UNKNOWN_CATEGORY"]`) | **Exact Fallback Enforcement**: Returns exact mandatory fallback message string without generating replacement text. |

---

## 4. Legal Record Validation Specification

Every legal record `L` passed to `validate_legal_record(item, db)` must satisfy all 5 criteria:
1. `item` is a valid dictionary containing non-empty `act`, `section`, and `description` strings.
2. Query `db.query(Law).filter(Law.act_name == act, Law.section_number == section)` returns at least 1 empirical row in PostgreSQL `laws`.
3. `item["description"]` matches `db_row.section_text` verbatim.
4. `item["act"]` matches `db_row.act_name` verbatim.
5. `item["section"]` matches `str(db_row.section_number)` verbatim.

If any criterion fails, the record is **REJECTED** and logged in audit rejection metrics.

---

## 5. Support Service Record Validation Specification

Every support service record `S` passed to `validate_support_record(item, db)` must satisfy all 5 criteria:
1. `item` is a valid dictionary containing non-empty `service_name`, `contact_number`, `state`, and `district` strings.
2. Query `db.query(SupportService).filter(SupportService.name == service_name, SupportService.contact_number == contact_number)` returns at least 1 empirical row in PostgreSQL `support_services`.
3. `item["state"]` matches `db_row.state` verbatim.
4. `item["district"]` matches `db_row.district` verbatim.
5. `item["service_type"]` matches `db_row.service_type` verbatim.

If any criterion fails, the record is **REJECTED** and logged in audit rejection metrics.

---

## 6. Multi-Label & Fallback Behavior

### Multi-Label Independent Validation
When multiple BERT categories are predicted (e.g. `["Domestic Violence", "Workplace Harassment"]`):
- Legal and support records for each category are validated independently.
- If `DV` produces valid DB records but `WH` produces 0 DB records, valid `DV` records are retained, and no synthetic `WH` information is generated.

### Mandatory Exact Fallback Messages

#### Legal Fallback (when 0 valid legal records exist):
```
Information not available in the provided knowledge base.
```

#### Support Fallback (when 0 valid support records exist):
```
No matching support service was found in the available support-services database.
```

---

## 7. Forbidden Sources Policy

The Anti-Hallucination Layer strictly prohibits:
- Google Search or Web Scraping
- External Legal APIs or Support Directories
- Hard-coded dictionary mappings of legal sections or phone numbers
- LLM generative text calls for legal provisions or support contacts

**Only the PostgreSQL database (`womens_safety_db`) is authoritative.**

---

## 8. Audit Metrics Structure

Internal validation logs details for monitoring and debugging:

```json
{
    "legal_matched": true,
    "legal_information": [...],
    "legal_message": null,
    "support_matched": true,
    "support_services": [...],
    "support_message": null,
    "audit": {
        "legal": {
            "received": 10,
            "accepted": 10,
            "rejected": 0,
            "rejection_reasons": []
        },
        "support": {
            "received": 6,
            "accepted": 6,
            "rejected": 0,
            "rejection_reasons": []
        }
    }
}
```

---

## 9. Hallucination Attack Test Suite Summary

All 13 test cases in [`backend/tests/test_hallucination.py`](file:///D:/Abhera(Mini)/backend/tests/test_hallucination.py) passed:

1. `test_01_valid_legal_record_is_accepted`: Valid DB legal record passes validation.
2. `test_02_valid_support_record_is_accepted`: Valid DB support record passes validation.
3. `test_03_fabricated_legal_record_is_rejected`: Fake legal section (`Fake Women's Safety Act, Sec 999`) **REJECTED**.
4. `test_04_fabricated_support_record_is_rejected`: Fake helpline (`Fake Women's Helpline, 9999999999`) **REJECTED**.
5. `test_05_unknown_category_does_not_generate_law`: `["UNKNOWN_CATEGORY"]` returns exact legal fallback message.
6. `test_06_unknown_category_does_not_generate_support`: `["UNKNOWN_CATEGORY"]` returns exact support fallback message.
7. `test_07_missing_legal_data_is_not_filled`: Missing description or section is **REJECTED** (no auto-fill).
8. `test_08_missing_support_contact_is_not_filled`: Missing phone number is **REJECTED** (no auto-fill).
9. `test_09_multiple_labels`: Multi-label payload validated independently.
10. `test_10_database_values_preserved`: Verifies verbatim DB value retention.
11. `test_11_no_external_source_dependency`: Verifies zero external API calls.
12. `test_12_empty_results`: Empty list inputs return safe exact fallbacks.
13. `test_13_explicit_hallucination_attack`: **SIMULATED MODEL ATTACK**: Injected fake laws & fake helplines resulted in 100% rejection (`rejected=1`) and exact fallback outputs.

---

## 10. System Limitations

- **Coverage Boundary**: Legal and support information is limited strictly to the 32 legal provisions in `laws` and 12 support services in `support_services`.
- **Future Expansion**: Adding new laws or support organizations requires executing a database migration/seed script to insert verified rows into PostgreSQL.
