"""
Diagnostic script for Multi-Label Legal Mapping Pipeline in ABHERA.
"""

import sys
import json
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.inference.predict_bert import BERTIncidentPredictor
from ml.inference.predict_ner import NEREntityPredictor
from app.database import SessionLocal
from app.services.legal_mapping import map_legal_provisions
from app.services.anti_hallucination import validate_payload
from app.services.report_generator import generate_incident_report

def run_diagnostic(test_name, text):
    print("=" * 80)
    print(f"DIAGNOSTIC CASE: {test_name}")
    print(f"INPUT TEXT: '{text}'")
    print("-" * 80)

    # 1. BERT Inference
    predictor = BERTIncidentPredictor()
    bert_res = predictor.predict(text)
    
    print("\nA. RAW BERT SCORES FOR EVERY LABEL:")
    for code, score in bert_res.get("all_scores", {}).items():
        thresh = bert_res.get("thresholds", {}).get(code, 0.50)
        status = "PASSED THRESHOLD" if score >= thresh else "below threshold"
        print(f"  - {code:5s}: score = {score:.4f} (threshold = {thresh}) -> {status}")

    print("\nB. FINAL SELECTED BERT LABELS:")
    pred_labels = bert_res.get("predicted_labels", [])
    for item in pred_labels:
        print(f"  - {item['code']}: {item['name']} (prob = {item['probability']})")
    
    label_names = [item["name"] for item in pred_labels]
    label_codes = [item["code"] for item in pred_labels]

    # 2. NER Extraction
    ner_pred = NEREntityPredictor()
    ner_entities = ner_pred.predict(text)
    print("\nC. NER ENTITIES:")
    if ner_entities:
        for ent in ner_entities:
            print(f"  - {ent.get('label')}: '{ent.get('text')}'")
    else:
        print("  - None extracted")

    # 3. Legal Mapping
    db = SessionLocal()
    try:
        print("\nD. LEGAL MAPPING INPUT LABELS:")
        print(f"  - Passed to map_legal_provisions: {label_names}")

        legal_res = map_legal_provisions(predicted_labels=label_names, db=db)
        print("\nE. EVERY LAW RETURNED BY legal_mapping.py:")
        legal_list = legal_res.get("legal_information", [])
        if legal_list:
            for idx, law in enumerate(legal_list, 1):
                print(f"  {idx}. Act: {law.get('act')} | Section: {law.get('section')} | Applicable Label: {law.get('applicable_label')}")
                print(f"     Text snippet: {law.get('description')[:80]}...")
        else:
            print(f"  - No laws returned. Message: {legal_res.get('message')}")

        # Check if laws for both DV and CA are returned
        returned_labels = set(law.get("applicable_label") for law in legal_list)
        print("\nF. RETURNED LAW CATEGORY LABELS IN LEGAL MAPPING:")
        print(f"  - Applicable Labels found in legal results: {returned_labels}")
        has_dv = "DV" in returned_labels
        has_ca = "CA" in returned_labels
        print(f"  - Contains Domestic Violence (DV) laws: {has_dv}")
        print(f"  - Contains Cyber Abuse (CA) laws: {has_ca}")

        # 4. Report Generator Payload Check
        report = generate_incident_report(
            narrative_text=text,
            bert_results=pred_labels,
            ner_entities=ner_entities,
            db=db,
        )

        print("\nG. REPORT GENERATOR RECEIVES ALL RETURNED LAWS:")
        report_laws = report.get("relevant_legal_information", [])
        if isinstance(report_laws, list):
            print(f"  - Total laws in report payload: {len(report_laws)}")
            report_labels = set(law.get("applicable_label") for law in report_laws)
            print(f"  - Applicable Labels in report payload: {report_labels}")
        else:
            print(f"  - Report legal field: {report_laws}")

        print("\nH. FRONTEND DISPLAY SIMULATION (normalizeReportData):")
        identified_types = report.get("identified_incident_type", {}).get("labels", [])
        print(f"  - Identified Incident Types in Report JSON: {identified_types}")
        
    finally:
        db.close()

    print("=" * 80 + "\n")


if __name__ == "__main__":
    primary_text = "My husband beats me and threatens to post my private photos online."
    run_diagnostic("PRIMARY DIAGNOSTIC CASE", primary_text)

    test_cases = [
        ("Case 1: Domestic Violence only", "My husband beats me."),
        ("Case 2: Cyber Abuse only", "Someone threatens to post my private photos online."),
        ("Case 3: DV + Cyber Abuse", "My husband beats me and threatens to post my private photos online."),
        ("Case 4: DV + Cyber Abuse (Threats)", "My husband beats me, threatens me, and threatens to post my private photos online."),
        ("Case 5: DV + Cyber Abuse + Unsafe", "My husband beats me and threatens to post my private photos online, and I am currently unsafe."),
    ]

    for name, txt in test_cases:
        run_diagnostic(name, txt)
