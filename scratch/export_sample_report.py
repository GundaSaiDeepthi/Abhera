import sys
import os
import json

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database import SessionLocal
from app.services.report_generator import ReportGeneratorService

def main():
    db = SessionLocal()
    service = ReportGeneratorService(db=db)

    # Sample realistic incident with NER, Legal, Support, Q&A
    report = service.generate_report(
        submission_id="SUB-2026-88E239AF",
        narrative_text="My husband and in-laws beat me and harass me for dowry every day at home. They have threatened to throw me out of the house if my parents do not pay 5 lakh rupees.",
        bert_results={"predicted_labels": ["Domestic Violence", "Dowry Harassment"]},
        ner_entities=[
            {"label": "PERP_REL", "text": "husband and in-laws"},
            {"label": "TIME_FREQ", "text": "every day"},
            {"label": "LOCATION", "text": "at home"},
            {"label": "EVIDENCE", "text": "threatened to throw me out"},
        ],
        user_answers=[
            {"question_id": "Q_SAFETY_01", "question": "Are you currently in immediate physical danger?", "answer": "No, I am at a neighbor's house right now."},
            {"question_id": "Q_CHILDREN_01", "question": "Are there any children residing with you who are affected?", "answer": "Yes, my 4-year-old daughter."},
            {"question_id": "Q_MEDICAL_01", "question": "Do you require medical assistance for any injuries?", "answer": "Yes, I have bruises on my arms and back."}
        ],
        state="Delhi",
        district="New Delhi",
        db_session=db
    )

    out_path = os.path.join(os.path.dirname(__file__), 'sample_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Sample report successfully exported to: {out_path}")
    db.close()

if __name__ == '__main__':
    main()
