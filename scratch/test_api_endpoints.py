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

print("=== START SESSION ===")
s_resp = client.post("/api/session/start")
s_data = s_resp.json()
session_id = s_data["session_id"]
submission_id = s_data["submission_id"]
print("session_id:", session_id)
print("submission_id:", submission_id)

print("\n=== TURN 1 ===")
r1 = client.post("/api/chat/message", json={"session_id": session_id, "message": "My husband beats me regularly and threatens me."})
d1 = r1.json()
print("Turn 1 status:", d1.get("conversation_status"))
print("Turn 1 report:", d1.get("report"))

print("\n=== TURN 2 ===")
r2 = client.post("/api/chat/message", json={"session_id": session_id, "message": "yes"})
d2 = r2.json()
print("Turn 2 status:", d2.get("conversation_status"))

print("\n=== TURN 3 ===")
r3 = client.post("/api/chat/message", json={"session_id": session_id, "message": "partner"})
d3 = r3.json()
print("Turn 3 status:", d3.get("conversation_status"))

print("\n=== TURN 4 ===")
r4 = client.post("/api/chat/message", json={"session_id": session_id, "message": "he beats me everyday after coming home"})
d4 = r4.json()
print("Turn 4 status:", d4.get("conversation_status"))
print("Turn 4 report present:", d4.get("report") is not None)

print("\n==================================================")
print("STEP 1 & 2: FINAL COMPLETED CHAT RESPONSE JSON")
print("==================================================")
chat_report = d4.get("report")
print("Full POST /api/chat/message response:")
print(json.dumps(d4, indent=2))

if chat_report:
    print("\n--- REPORT SECTIONS IN CHAT RESPONSE ---")
    print("report.identified_incident_type:", json.dumps(chat_report.get("identified_incident_type"), indent=2))
    print("report.extracted_information:", json.dumps(chat_report.get("extracted_information"), indent=2))
    print("report.relevant_legal_information count:", len(chat_report.get("relevant_legal_information", [])))
    print("report.available_support_services count:", len(chat_report.get("available_support_services", [])))
else:
    print("WARNING: report is None in final chat response!")

print("\n==================================================")
print("STEP 6: GET /api/report/{submission_id} RESPONSE")
print("==================================================")
get_r = client.get(f"/api/report/{submission_id}")
print("GET Status Code:", get_r.status_code)
get_report = get_r.json()
print(json.dumps(get_report, indent=2))

print("\n==================================================")
print("COMPARISON CHECK:")
print("==================================================")
print("Chat Report identified_incident_type:", chat_report.get("identified_incident_type") if chat_report else None)
print("GET Report identified_incident_type:", get_report.get("identified_incident_type") if get_report else None)
print("Equal?", chat_report == get_report)

# Clean up SQLite test file
try:
    if os.path.exists("abhera_test.db"):
        os.remove("abhera_test.db")
except Exception:
    pass
