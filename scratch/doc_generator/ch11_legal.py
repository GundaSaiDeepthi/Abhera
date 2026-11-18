"""
Module for Chapter 11: Legal Guidance and Report Generation
"""

def get_chapter_11():
    return r'''# CHAPTER 11: Legal Guidance and Report Generation

---

## 11.1 The Anti-Hallucination Gating Philosophy

The most dangerous pitfall in applying artificial intelligence to safety-critical legal aid is **hallucination**. 

Modern generative Large Language Models (LLMs) operate on token probabilities. When asked for legal advice, an LLM will often invent non-existent penal code sections (e.g., citing *"Section 420A BNS for WhatsApp harassment"*), fabricate prison sentences, or confuse civil tort remedies with cognizable criminal offenses. If a survivor cites a fabricated law in an official police First Information Report (FIR), the complaint can be dismissed immediately for lack of legal foundation, leaving the survivor unprotected.

In ABHERA, the incident report is assembled using a **deterministic, non-generative document compiler**:
$$\\text{Report Data} = \\text{Survivor Statements} + \\text{Direct Verified PostgreSQL Rows}$$

```
+---------------------------------------------------------------------------------------------------+
|                                 THE ABHERA ZERO-HALLUCINATION MOTTO                               |
+---------------------------------------------------------------------------------------------------+
|  "The model predicts and extracts; the database verifies and provides the final legal and         |
|   support information."                                                                           |
|                                                                                                   |
|  Every legal section number, statutory definition, and emergency helpline contact in the final    |
|  report originates strictly and verbatim from audited PostgreSQL database rows.                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 11.2 Legal Mapping Engine (`backend/app/services/legal_mapping.py`)

The Legal Mapping Service translates predicted machine learning category codes into official statutory provisions stored in the PostgreSQL `laws` table:

```python
# Label Normalization Map in legal_mapping.py
LABEL_TO_APPLICABLE_CODE = {
    "DV": "DV", "DOMESTIC VIOLENCE": "DV",
    "SH": "SH", "SEXUAL HARASSMENT": "SH",
    "ST": "ST", "STALKING": "ST",
    "CA": "CA", "CYBER HARASSMENT": "CA", "CYBER ABUSE": "CA",
    "WH": "WH", "WORKPLACE HARASSMENT": "WH",
    "OV": "OV", "OTHER HARASSMENT": "OV",
    "FPM": "FPM", "FALSE PROMISE OF MARRIAGE": "FPM",
    "DOWRY": "DOWRY", "DOWRY HARASSMENT": "DOWRY",
}
```

### The 32 Audited Indian Legal Provisions in the Knowledge Base:
The `laws` table contains exactly 32 verified provisions across five key Indian statutes:
1. **Bharatiya Nyaya Sanhita (BNS 2023)**:
   - *Section 69*: Sexual intercourse by employing deceitful means (including false promise of employment or marriage).
   - *Section 74*: Assault or use of criminal force to woman with intent to outrage her modesty.
   - *Section 75*: Sexual harassment (unwelcome physical contact, sexually colored remarks, pornography demand).
   - *Section 76*: Assault or use of criminal force to woman with intent to disrobe.
   - *Section 77*: Voyeurism (watching or capturing images of a woman in private acts).
   - *Section 78*: Stalking (following a woman or monitoring her internet/electronic communication).
   - *Section 79*: Word, gesture, or act intended to insult the modesty of a woman.
   - *Section 85*: Husband or relative of husband of a woman subjecting her to cruelty.
   - *Section 86*: Definition of cruelty (mental or physical harm, coercion for dowry).
   - *Section 351*: Criminal intimidation.
2. **Information Technology Act, 2000 (IT Act)**:
   - *Section 66E*: Violation of privacy (intentionally capturing or transmitting images of private body areas without consent).
   - *Section 67*: Punishment for publishing or transmitting obscene material in electronic form.
   - *Section 67A*: Punishment for publishing or transmitting material containing sexually explicit acts in electronic form.
3. **Protection of Women from Domestic Violence Act, 2005 (PWDVA)**:
   - *Section 3*: Broad definition of domestic violence (physical, sexual, verbal, emotional, economic).
   - *Section 12*: Application to Magistrate for protection orders, residence orders, and monetary relief.
   - *Section 18*: Protection orders prohibiting perpetrator from committing acts of domestic violence or entering place of employment.
   - *Section 19*: Residence orders prohibiting dispossession of the survivor from the shared household.
   - *Section 20 & 22*: Monetary relief and compensation orders.
4. **Sexual Harassment of Women at Workplace (POSH Act 2013)**:
   - *Section 3*: Prevention of sexual harassment at workplace.
   - *Section 4*: Constitution of Internal Complaints Committee (ICC) in every organization with 10+ employees.
   - *Section 9*: Complaint of sexual harassment filing procedure (within 3 months of incident).
   - *Section 13*: Inquiry report and interim relief measures (transfer, leave).
5. **Dowry Prohibition Act, 1961**:
   - *Section 3 & 4*: Penalty for giving, taking, or demanding dowry.

---

## 11.3 Emergency Support Service Routing (`backend/app/services/support_mapping.py`)

ABHERA routes survivors to verified support infrastructure organized across a **3-tier directory** in the `support_services` table:

```
+===================================================================================================+
|                                 ABHERA 3-TIER EMERGENCY ROUTING                                   |
+------+-----------------------+-----------------------------+--------------------------------------+
| Tier | Level                 | Service Name                | Contact Number / Channel             |
+------+-----------------------+-----------------------------+--------------------------------------+
| 1    | National Immediate    | National Emergency Helpline | 112 (Toll-Free, 24/7 Police/Medical) |
|      |                       | Women Helpline (National)   | 181 (Toll-Free, 24/7 Crisis Support) |
|      |                       | Women Power Line            | 1090 (Harassment / Stalking)         |
+------+-----------------------+-----------------------------+--------------------------------------+
| 2    | Specialized Digital   | Cybercrime Reporting Portal | cybercrime.gov.in / 1930             |
|      |                       | NCW Complaint Cell          | 7827170170 (National Women Comm.)   |
+------+-----------------------+-----------------------------+--------------------------------------+
| 3    | State & Local         | Sakhi One Stop Centres (OSC)| District Integrated Crisis Units     |
|      |                       | Legal Aid Defense Counsel   | State Legal Services Authority (DLSA)|
+------+-----------------------+-----------------------------+--------------------------------------+
```

---

## 11.4 The 10-Section Structured Report Schema

The formal incident report compiled by `ReportGeneratorService` contains exactly **10 immutable sections**:

```
+---------------------------------------------------------------------------------------------------+
|                              THE 10-SECTION REPORT DOSSIER SCHEMA                                 |
+---------------------------------------------------------------------------------------------------+
| Section 1: Incident Summary           | Unaltered chronological statement provided by the survivor.|
+---------------------------------------+-----------------------------------------------------------+
| Section 2: Identified Incident Type   | Verified harassment categories predicted by BERT (e.g. ST)|
+---------------------------------------+-----------------------------------------------------------+
| Section 3: Extracted Information (NER)| Structured entities: Perpetrator, Location, Platform, Time|
+---------------------------------------+-----------------------------------------------------------+
| Section 4: User-Provided Details (Q&A)| Factual answers provided during dynamic follow-up dialogue|
+---------------------------------------+-----------------------------------------------------------+
| Section 5: Relevant Legal Information | Audited statutory sections retrieved from PostgreSQL laws |
+---------------------------------------+-----------------------------------------------------------+
| Section 6: Suggested Next Steps       | Practical safety, legal, and evidence-preservation steps   |
+---------------------------------------+-----------------------------------------------------------+
| Section 7: Available Support Services | Verified crisis hotlines, cyber portals, and local OSCs   |
+---------------------------------------+-----------------------------------------------------------+
| Section 8: Evidence / Info Notes      | Catalog of mentioned screenshots, witnesses, and chats    |
+---------------------------------------+-----------------------------------------------------------+
| Section 9: Model Prediction Metadata  | Technical audit trail: model names, tasks, data provenance|
+---------------------------------------+-----------------------------------------------------------+
| Section 10: Legal & Medical Disclaimer| Mandatory statutory disclaimer regarding legal advice     |
+---------------------------------------------------------------------------------------------------+
```

---

## 11.5 Anti-Hallucination Service Implementation (`backend/app/services/anti_hallucination.py`)

The `AntiHallucinationService` functions as an impenetrable firewall between the AI predictions and the final response.

```python
class AntiHallucinationService:
    def validate_legal_record(self, item: Dict[str, Any], db: DBSession) -> Tuple[bool, Optional[str]]:
        """Asserts that a legal item matches an empirical row in PostgreSQL verbatim."""
        act = item.get("act")
        section = item.get("section")
        
        # Query PostgreSQL laws table for exact matching record
        db_match = (
            db.query(Law)
            .filter(
                Law.act_name == str(act).strip(),
                Law.section_number == str(section).strip(),
            )
            .first()
        )
        if not db_match:
            return False, f"Legal provision '{act} {section}' not found in verified database."
        return True, None
```

If an unmapped label or an AI-generated hallucination attempts to enter the report, the Anti-Hallucination layer rejects it and substitutes the mandatory fallback:
> *"No specific statutory provisions could be automatically determined from the narrative. Please consult a legal professional or women's helpline for personalized legal advice."*

---

## 11.6 Legal and Ethical Disclaimers

ABHERA operates strictly within ethical AI guidelines:
1. **Informational Triage Only**: ABHERA is not a law firm, does not act as an attorney, and does not provide formal legal advice or legal representation.
2. **Emergency Disclaimer**: The platform explicitly informs survivors that if they are in immediate physical danger, they should exit the application and dial **112** immediately.
3. **No Automated Court Submissions**: The report is provided to the survivor as an informational PDF for their own personal record or to hand to their chosen legal counsel or police station.
'''
