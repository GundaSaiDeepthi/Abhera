import csv
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
from ml.inference.predict_bert import BERTIncidentPredictor
from ml.inference.predict_ner import NEREntityPredictor

# Create tables in sqlite test db
Base.metadata.create_all(bind=engine)
db = SessionLocal()

# Seed questions
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

print("==================================================")
print("1. DIRECT BERT PREDICTION TEST")
print("==================================================")
bert = BERTIncidentPredictor()
t1 = "my husband beats me everyday"
t2 = "my husband beats me everyday. he beats me everyday after coming home...he also drinks alcohol"

res_bert1 = bert.predict(t1)
res_bert2 = bert.predict(t2)

print("Text 1:", t1)
print("Predicted Labels:", res_bert1.get("predicted_labels"))
print("DV Score:", res_bert1.get("all_scores", {}).get("DV"), "(Threshold: 0.55)")

print("\nText 2:", t2)
print("Predicted Labels:", res_bert2.get("predicted_labels"))

print("\n==================================================")
print("2. DIRECT NER PREDICTION TEST")
print("==================================================")
ner = NEREntityPredictor()
res_ner1 = ner.predict(t1)
res_ner2 = ner.predict(t2)
print("NER Text 1:", res_ner1)
print("NER Text 2:", res_ner2)

print("\n==================================================")
print("3. LIVE END-TO-END MULTI-TURN CHAT & REPORT API TEST")
print("==================================================")
client = TestClient(app)

# Step A: Start session
start_resp = client.post("/api/session/start")
assert start_resp.status_code == 200, f"Session start failed: {start_resp.text}"
session_data = start_resp.json()
session_id = session_data["session_id"]
submission_id = session_data["submission_id"]
print("Created Session ID:", session_id)
print("Created Submission ID:", submission_id)

# Step B: Turn 1 - Initial Narrative
msg1_resp = client.post("/api/chat/message", json={"session_id": session_id, "message": "my husband beats me everyday"})
assert msg1_resp.status_code == 200, f"Turn 1 failed: {msg1_resp.text}"
data1 = msg1_resp.json()
print("\nTurn 1 Assistant Message:", data1.get("assistant_message"))
print("Turn 1 Predicted Labels:", data1.get("predicted_labels"))
print("Turn 1 Entities:", data1.get("entities"))
print("Turn 1 Status:", data1.get("conversation_status"))

# Step C: Turn 2 - Follow-up answer "yes"
msg2_resp = client.post("/api/chat/message", json={"session_id": session_id, "message": "yes"})
assert msg2_resp.status_code == 200, f"Turn 2 failed: {msg2_resp.text}"
data2 = msg2_resp.json()
print("\nTurn 2 Assistant Message:", data2.get("assistant_message"))
print("Turn 2 Predicted Labels:", data2.get("predicted_labels"))
print("Turn 2 Entities:", data2.get("entities"))
print("Turn 2 Status:", data2.get("conversation_status"))

# Step D: Turn 3 - Follow-up answer "partner"
msg3_resp = client.post("/api/chat/message", json={"session_id": session_id, "message": "partner"})
assert msg3_resp.status_code == 200, f"Turn 3 failed: {msg3_resp.text}"
data3 = msg3_resp.json()
print("\nTurn 3 Assistant Message:", data3.get("assistant_message"))
print("Turn 3 Predicted Labels:", data3.get("predicted_labels"))
print("Turn 3 Entities:", data3.get("entities"))
print("Turn 3 Status:", data3.get("conversation_status"))

# Step E: Turn 4 - Narrative update
msg4_text = "he beats me everyday after coming home...he also drinks alcohol"
msg4_resp = client.post("/api/chat/message", json={"session_id": session_id, "message": msg4_text})
assert msg4_resp.status_code == 200, f"Turn 4 failed: {msg4_resp.text}"
data4 = msg4_resp.json()
print("\nTurn 4 Assistant Message:", data4.get("assistant_message"))
print("Turn 4 Predicted Labels:", data4.get("predicted_labels"))
print("Turn 4 Entities:", data4.get("entities"))
print("Turn 4 Status:", data4.get("conversation_status"))

# Step F: Retrieve final generated report
report = data4.get("report")
if not report:
    rep_resp = client.get(f"/api/report/{submission_id}")
    assert rep_resp.status_code == 200, f"Get report failed: {rep_resp.text}"
    report = rep_resp.json()

print("\n==================================================")
print("4. FINAL INCIDENT REPORT VERIFICATION")
print("==================================================")
print("Incident Summary:", report.get("incident_summary"))
print("Identified Incident Type:", report.get("identified_incident_type"))
print("Extracted Information (NER):", report.get("extracted_information"))
print("User-Provided Details:", report.get("user_provided_details"))
print("Relevant Legal Information Count:", len(report.get("relevant_legal_information", [])))
print("First Legal Provision:", report.get("relevant_legal_information", [])[0] if report.get("relevant_legal_information") else None)
print("Available Support Services Count:", len(report.get("available_support_services", [])))
print("First Support Service:", report.get("available_support_services", [])[0] if report.get("available_support_services") else None)
print("Disclaimer:", report.get("disclaimer")[:60] + "...")

# Clean up SQLite test file
try:
    if os.path.exists("abhera_test.db"):
        os.remove("abhera_test.db")
except Exception:
    pass
