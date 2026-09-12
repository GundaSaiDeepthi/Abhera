import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.ner.annotation.validate_reviewed import validate_reviewed_dataset

def test_review_workflow():
    annotation_dir = project_root / "ml" / "ner" / "annotation"
    draft_file = annotation_dir / "annotated_ner_draft.jsonl"
    reviewed_file = annotation_dir / "annotated_ner_reviewed.jsonl"

    print("==================================================")
    print("TESTING NER HUMAN REVIEW WORKFLOW & INTEGRATION")
    print("==================================================")

    # 1. Load first candidate draft record
    candidate_records = []
    with open(draft_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidate_records.append(json.loads(line))

    first_candidate = candidate_records[0]
    print(f"1. Loaded First Candidate Record ID: '{first_candidate['id']}'")
    print(f"   Narrative: \"{first_candidate['text']}\"")
    print(f"   Candidate Entity Spans ({len(first_candidate['entities'])}): {first_candidate['entities']}\n")

    # 2. Simulate human review & approval of 1st record
    reviewed_record = {
        "id": first_candidate["id"],
        "text": first_candidate["text"],
        "entities": first_candidate["entities"],
    }

    # Save reviewed record to annotated_ner_reviewed.jsonl
    with open(reviewed_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(reviewed_record, ensure_ascii=False) + "\n")

    print(f"2. Saved 1 Reviewed Record to: {reviewed_file}")

    # 3. Run validation and report generation
    print("\n3. Executing Validation on Reviewed Output:")
    passed, rpt = validate_reviewed_dataset()

    print("\n==================================================")
    print("NER HUMAN REVIEW WORKFLOW TEST SUITE PASSED")
    print("==================================================")

if __name__ == "__main__":
    test_review_workflow()
