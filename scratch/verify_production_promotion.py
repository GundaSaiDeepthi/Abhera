import json
from pathlib import Path
import sys

# Ensure backend and root are in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from ml.inference.predict_bert import BERTIncidentPredictor
from app.database import SessionLocal
from app.services.legal_mapping import map_legal_provisions
from app.services.support_mapping import map_support_services
from app.question_engine.rules import get_required_information
from app.services.report_generator import generate_incident_report

TEST_CASES = [
    {
        "category": "DV (Domestic Violence)",
        "narrative": "My husband physically assaulted me at home, slapped me during an argument, and locked me inside the room."
    },
    {
        "category": "SH (Sexual Harassment)",
        "narrative": "A man at the bus stop made obscene gestures, touched my shoulder without consent, and whispered vulgar comments."
    },
    {
        "category": "ST (Cyber Stalking / Physical Stalking)",
        "narrative": "A stranger has been following me home every night and calling me continuously from hidden numbers."
    },
    {
        "category": "CA (Cyber Abuse)",
        "narrative": "Someone created a fake profile with my photos online and is sending abusive messages to my friends."
    },
    {
        "category": "WH (Workplace Harassment)",
        "narrative": "My senior manager at the office repeatedly sends inappropriate messages after work hours and threatened to block my promotion."
    },
    {
        "category": "OV (Online Violent Threats / Street Abuse)",
        "narrative": "A violent group of men blocked my path on the street, threatened me with physical harm, and threw stones at my car."
    },
    {
        "category": "Multi-Label Scenario 1 (DV + CA)",
        "narrative": "My husband physically abuses me at home and threatens to post my private photos online."
    },
    {
        "category": "Multi-Label Scenario 2 (WH + SH)",
        "narrative": "My boss at the office touches me inappropriately during work meetings and insists I meet him outside work for a promotion."
    },
    {
        "category": "Multi-Label Scenario 3 (ST + CA)",
        "narrative": "An anonymous stalker follows me every day after college and posts abusive threats on social media."
    }
]


def run_promotion_verification():
    db = SessionLocal()
    predictor = BERTIncidentPredictor()

    print("==================================================")
    print("PRODUCTION BERT V2 INTEGRATION & DOWNSTREAM AUDIT")
    print("==================================================")

    results_summary = []

    for idx, test_case in enumerate(TEST_CASES, start=1):
        cat = test_case["category"]
        text = test_case["narrative"]

        print(f"\n--------------------------------------------------")
        print(f"CASE {idx}: [{cat}]")
        print(f"Narrative: \"{text}\"")

        # 1. BERT Inference
        pred_res = predictor.predict(text)
        pred_labels = pred_res["predicted_labels"]
        pred_names = [p["name"] for p in pred_labels]
        pred_codes = [p["code"] for p in pred_labels]

        print(f"Predicted Labels: {pred_names} (Codes: {pred_codes})")

        # 2. Legal Mapping
        legal_res = map_legal_provisions(predicted_labels=pred_names, db=db)
        legal_matched = legal_res["matched"]
        legal_provisions_count = len(legal_res.get("legal_information", []))
        print(f"Legal Mapping: Matched = {legal_matched} | Provisions Found = {legal_provisions_count}")

        # 3. Support Mapping
        support_res = map_support_services(predicted_labels=pred_names, db=db)
        support_matched = support_res["matched"]
        support_services_count = len(support_res.get("support_services", []))
        print(f"Support Mapping: Matched = {support_matched} | Services Found = {support_services_count}")

        # 4. Question Engine Requirement Determination
        required_info = get_required_information(pred_names)
        questions_count = len(required_info)
        print(f"Question Engine: Triggered {questions_count} required information fields {required_info}")

        # 5. Mock Report Generation Check
        mock_submission_data = {
            "submission_id": f"SUB-VERIFY-{idx:03d}",
            "incident_type": ", ".join(pred_names),
            "narrative_text": text,
            "entities": [{"text": "Perpetrator", "label": "PERP"}],
            "question_answers": []
        }
        report_res = generate_incident_report(
            submission_data=mock_submission_data,
            legal_info=legal_res.get("legal_information", []),
            support_info=support_res.get("support_services", [])
        )
        report_generated = bool(report_res and "report_content" in report_res)
        print(f"Report Generation: Success = {report_generated}")

        results_summary.append({
            "case_id": idx,
            "category": cat,
            "narrative": text,
            "predicted_labels": pred_names,
            "predicted_codes": pred_codes,
            "legal_matched": legal_matched,
            "support_matched": support_matched,
            "questions_triggered": questions_count,
            "report_generated": report_generated
        })

    db.close()

    out_file = Path("promotion_results.json").resolve()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"SUCCESS: RESULTS SAVED TO {out_file}")


if __name__ == "__main__":
    run_promotion_verification()
