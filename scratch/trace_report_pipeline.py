import csv
import json
import os
import sys
from pathlib import Path

# Setup SQLite for test execution before app import
os.environ["DATABASE_URL"] = "sqlite:///./abhera_test.db"

project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from app.database import Base, engine, SessionLocal
from app.models.law import Law
from app.models.support_service import SupportService
from app.db.seed_questions import seed_questions
from fastapi.testclient import TestClient
from app.main import app

# Create tables in sqlite test db and seed data
Base.metadata.create_all(bind=engine)
db = SessionLocal()
seed_questions(db)

# Seed laws from data/laws.csv
laws_csv_path = project_root / "data" / "laws.csv"
if laws_csv_path.exists():
    with open(laws_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            law_obj = Law(
                act_name=row["act_name"].strip(),
                section_number=row["section_number"].strip(),
                section_text=row["section_text"].strip(),
                applicable_label=row["applicable_label"].strip(),
            )
            db.add(law_obj)

# Seed support services from data/support_services.csv
supp_csv_path = project_root / "data" / "support_services.csv"
if supp_csv_path.exists():
    with open(supp_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            supp_obj = SupportService(
                service_type=row["service_type"].strip(),
                name=row["name"].strip(),
                contact_number=row["contact_number"].strip(),
                state=row["state"].strip(),
                district=row["district"].strip(),
                applicable_label=row["applicable_label"].strip(),
            )
            db.add(supp_obj)

db.commit()
db.close()

client = TestClient(app)

print("==================================================")
print("STEP 1: CREATE SESSION AND RUN MULTI-TURN CHAT")
print("==================================================")
start_resp = client.post("/api/session/start")
sess_data = start_resp.json()
session_id = sess_data["session_id"]
submission_id = sess_data["submission_id"]
print("Session ID:", session_id)
print("Submission ID:", submission_id)

turn_num = 1
msg_text = "My husband beats me regularly and threatens me."

while True:
    print(f"\n--- Turn {turn_num}: Sending '{msg_text}' ---")
    resp = client.post("/api/chat/message", json={"session_id": session_id, "message": msg_text})
    assert resp.status_code == 200, f"Turn {turn_num} failed: {resp.text}"
    resp_data = resp.json()
    
    print("Status:", resp_data.get("conversation_status"))
    print("Predicted Labels:", resp_data.get("predicted_labels"))
    print("Entities:", resp_data.get("entities"))
    print("Current Question:", resp_data.get("current_question", {}).get("question_text") if resp_data.get("current_question") else "None")
    
    if resp_data.get("conversation_status") == "COMPLETED":
        print("\n==================================================")
        print("STEP 2: FINAL COMPLETED CHAT RESPONSE JSON")
        print("==================================================")
        print(json.dumps(resp_data, indent=2))
        
        report_obj = resp_data.get("report")
        if report_obj:
            print("\n--- REPORT SECTIONS IN CHAT RESPONSE ---")
            print("report.identified_incident_type:", report_obj.get("identified_incident_type"))
            print("report.extracted_information:", report_obj.get("extracted_information"))
            print("report.relevant_legal_information count:", len(report_obj.get("relevant_legal_information", [])))
            print("report.available_support_services count:", len(report_obj.get("available_support_services", [])))
        else:
            print("WARNING: report_obj is None in chat response!")
        break
    
    turn_num += 1
    # Simple answers to complete questions
    if turn_num == 2:
        msg_text = "yes"
    elif turn_num == 3:
        msg_text = "partner"
    else:
        msg_text = "He threatens me almost everyday"

print("\n==================================================")
print("STEP 6: GET /api/report/{submission_id}")
print("==================================================")
get_rep_resp = client.get(f"/api/report/{submission_id}")
print("GET Status Code:", get_rep_resp.status_code)
get_rep_data = get_rep_resp.json()
print("GET report.identified_incident_type:", get_rep_data.get("identified_incident_type"))
print("GET report.extracted_information:", get_rep_data.get("extracted_information"))
print("GET report.relevant_legal_information count:", len(get_rep_data.get("relevant_legal_information", [])))
print("GET report.available_support_services count:", len(get_rep_data.get("available_support_services", [])))

# Clean up SQLite test file
try:
    if os.path.exists("abhera_test.db"):
        os.remove("abhera_test.db")
except Exception:
    pass
