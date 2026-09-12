import sys
import json
from pathlib import Path

sys.path.insert(0, r"d:\Abhera(Mini)")
sys.path.insert(0, r"d:\Abhera(Mini)\backend")

from app.database import SessionLocal
from app.services.chat_service import ChatService
from app.services.session_service import SessionService
from app.models.report import Report
from app.services.report_generator import ReportGeneratorService
from ml.inference.predict_ner import NEREntityPredictor

def test_dynamic_questions_and_persistence():
    db = SessionLocal()
    
    print("==================================================")
    print("SECTION 1: NER INSTAGRAM TEST")
    print("==================================================")
    ner = NEREntityPredictor()
    ner_res = ner.predict("My husband keeps sending threatening messages on Instagram.")
    print("NER Result for Instagram prompt:")
    for ent in ner_res:
        print(f"  - Label: {ent['label']:<12} Text: '{ent['text']}' (start: {ent['start_char']}, end: {ent['end_char']})")
    
    test_cases = [
        ("TEST 1 (Short DV)", "My husband beats me.", ["No immediate danger."]),
        ("TEST 2 (Detailed DV)", "My husband beats me regularly and threatens me.", ["No, I am safe."]),
        ("TEST 3 (College Stalking)", "Someone keeps following me near my college.", ["Every day after class.", "No immediate danger."]),
        ("TEST 4 (Instagram Cyber)", "Someone is sending threatening messages on Instagram.", ["It happens daily."]),
        ("TEST 5 (Colleague Workplace)", "My colleague is repeatedly harassing me at work.", ["At my office.", "No immediate physical danger."]),
    ]
    
    print("\n==================================================")
    print("SECTION 4, 5, 6, 7: DYNAMIC QUESTION ENGINE & REPETITION TEST")
    print("==================================================")
    
    summary_results = {}
    
    for test_id, initial_msg, answers in test_cases:
        print(f"\n--- {test_id} ---")
        print(f"Initial Narrative: '{initial_msg}'")
        sess = SessionService.create_session(db)
        
        # Turn 1
        res = ChatService.process_message(db, sess.session_id, initial_msg)
        labels = res.get("predicted_labels", [])
        entities = res.get("entities", {})
        q1 = res.get("current_question")
        q1_text = q1.get("question_text") if q1 else "None (Direct to Report)"
        q1_id = q1.get("question_id") if q1 else None
        
        print(f"Turn 1 -> Predicted Labels: {labels}")
        print(f"Turn 1 -> Extracted Entities: {entities}")
        print(f"Turn 1 -> Selected Question: {q1_id}: '{q1_text}'")
        
        asked_q_ids = [q1_id] if q1_id else []
        
        # Subsequent turns
        turn = 2
        for ans in answers:
            if res.get("conversation_status") in ["PROCESSING", "COMPLETED"]:
                break
            print(f"Turn {turn} -> User Answer: '{ans}'")
            res = ChatService.process_message(db, sess.session_id, ans)
            q_next = res.get("current_question")
            if q_next:
                q_next_id = q_next.get("question_id")
                q_next_text = q_next.get("question_text")
                print(f"Turn {turn} -> Selected Question: {q_next_id}: '{q_next_text}'")
                asked_q_ids.append(q_next_id)
            else:
                print(f"Turn {turn} -> Selected Question: None (Completed)")
            turn += 1
            
        print(f"All Asked Question IDs: {asked_q_ids}")
        has_duplicates = len(asked_q_ids) != len(set(asked_q_ids))
        print(f"Question Repetition Check (No Duplicates): {'PASS' if not has_duplicates else 'FAIL (DUPLICATES FOUND)'}")
        
        # SECTION 7: Narrative preservation check
        db_sess = SessionService.get_session(db, sess.session_id)
        first_msg = db_sess.messages[0]["content"] if db_sess.messages else None
        original_preserved = (first_msg == initial_msg)
        user_answers_stored = db_sess.answers
        print(f"Original Description Preserved: {original_preserved} ('{first_msg}')")
        print(f"Follow-up Answers Stored: {user_answers_stored}")
        
        # SECTION 8 & 9: Report Generation & Persistence Verification
        report = res.get("report")
        sub_id = report.get("submission_id") if report else None
        
        summary_results[test_id] = {
            "labels": labels,
            "entities": entities,
            "q1": q1_text,
            "asked_questions": asked_q_ids,
            "no_duplicates": not has_duplicates,
            "original_preserved": original_preserved,
            "report_generated": report is not None,
            "sub_id": sub_id,
        }
        
    # Close session to test persistence
    db.close()
    
    print("\n==================================================")
    print("SECTION 9: REPORT PERSISTENCE (GET /api/report/{sub_id})")
    print("==================================================")
    db_fresh = SessionLocal()
    for test_id, data in summary_results.items():
        sub_id = data["sub_id"]
        if not sub_id:
            print(f"{test_id}: NO SUBMISSION ID")
            continue
        report_obj = db_fresh.query(Report).filter(Report.submission_id == sub_id).first()
        fetched_report = report_obj.report_data if report_obj else None
        if fetched_report:
            print(f"PASS - Fetched {sub_id} for {test_id}:")
            print(f"  - Summary: '{fetched_report.get('incident_summary')[:60]}...'")
            print(f"  - Type: {fetched_report.get('identified_incident_type')}")
            print(f"  - Entities: {fetched_report.get('extracted_information')}")
            print(f"  - Legal Count: {len(fetched_report.get('relevant_legal_information', []))}")
            print(f"  - Support Count: {len(fetched_report.get('available_support_services', []))}")
        else:
            print(f"FAIL - Could not fetch report for {sub_id}")
    db_fresh.close()

if __name__ == "__main__":
    test_dynamic_questions_and_persistence()
