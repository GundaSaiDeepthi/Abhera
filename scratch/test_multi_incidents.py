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

bert = BERTIncidentPredictor()
ner = NEREntityPredictor()
client = TestClient(app)

test_descriptions = [
    "My husband beats me regularly and threatens me.",
    "Someone keeps following me and sends threatening messages.",
    "Someone created a fake social media account using my photos and is harassing me.",
    "My colleague keeps sending inappropriate messages and touching me at work.",
    "My in-laws demand money and threaten me because I cannot provide dowry."
]

print("================================================================================")
print("MULTI-INCIDENT DYNAMIC PIPELINE ANALYSIS")
print("================================================================================")

for idx, desc in enumerate(test_descriptions, 1):
    print(f"\n--------------------------------------------------------------------------------")
    print(f"TEST {idx}: {desc}")
    print(f"--------------------------------------------------------------------------------")
    
    # 1. BERT Analysis
    bert_res = bert.predict(desc)
    raw_scores = bert_res.get("all_scores", {})
    predicted_labels = bert_res.get("predicted_labels", [])
    label_names = [l["name"] for l in predicted_labels]
    
    print("BERT Raw Scores:", raw_scores)
    print("BERT Final Labels:", label_names)
    
    # 2. NER Analysis
    ner_entities = ner.predict(desc)
    print("NER Entities Extracted:", ner_entities)
    
    # 3. Live API Chat & Report Generation
    start_resp = client.post("/api/session/start")
    sess_data = start_resp.json()
    s_id = sess_data["session_id"]
    sub_id = sess_data["submission_id"]
    
    msg_resp = client.post("/api/chat/message", json={"session_id": s_id, "message": desc})
    chat_data = msg_resp.json()
    
    print("Question Selected:", chat_data.get("current_question", {}).get("question_text") if chat_data.get("current_question") else "Completed / No Question")
    
    # Generate final report for submission
    rep_resp = client.post("/api/report/generate", json={"submission_id": sub_id})
    report = rep_resp.json()
    
    legal_records = report.get("relevant_legal_information", [])
    support_records = report.get("available_support_services", [])
    
    print("Identified Incident Type:", report.get("identified_incident_type"))
    print("Extracted Information (NER):", report.get("extracted_information"))
    print("Legal Mapping Record Count:", len(legal_records) if isinstance(legal_records, list) else 0)
    print("Support Mapping Record Count:", len(support_records) if isinstance(support_records, list) else 0)

# Clean up SQLite test file
try:
    if os.path.exists("abhera_test.db"):
        os.remove("abhera_test.db")
except Exception:
    pass
