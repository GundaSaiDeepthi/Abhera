import json
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FixDraftRecords")


def fix_malformed_draft_records():
    project_root = Path(__file__).resolve().parent.parent.parent
    annotation_dir = project_root / "ml" / "ner" / "annotation"

    sample_jsonl = annotation_dir / "annotation_sample.jsonl"
    draft_jsonl = annotation_dir / "annotated_ner_draft.jsonl"
    report_file = annotation_dir / "draft_annotation_report.txt"

    # 1. Inspect original narrative text for WSL_0698 and WSL_1033 from sample JSONL
    sample_map = {}
    with open(sample_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                sample_map[rec["id"]] = rec["text"]

    target_ids = ["WSL_0698", "WSL_1033"]
    for tid in target_ids:
        logger.info(f"Sample Narrative for '{tid}': \"{sample_map.get(tid, 'NOT FOUND')}\"")

    # 2. Read draft JSONL, find WSL_0698 & WSL_1033, and ensure entities is []
    draft_records = []
    fixed_count = 0

    with open(draft_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_id = rec["id"]

            if rec_id in target_ids:
                # Ensure entities is empty list []
                rec["entities"] = []
                rec["text"] = sample_map.get(rec_id, rec["text"])
                fixed_count += 1
                logger.info(f"Fixed record '{rec_id}': set entities = []")

            draft_records.append(rec)

    # 3. Rewrite updated draft JSONL
    with open(draft_jsonl, "w", encoding="utf-8") as f:
        for rec in draft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved corrected draft JSONL to: {draft_jsonl}")


if __name__ == "__main__":
    fix_malformed_draft_records()
