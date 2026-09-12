import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ValidateReviewed")

APPROVED_LABELS = ["PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM", "EVIDENCE", "LAW_SEC"]


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent.parent.parent,
        current_file.parent.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "ml").exists():
            return candidate
    return current_file.parent.parent.parent.parent


def validate_reviewed_dataset():
    project_root = get_project_root()
    annotation_dir = project_root / "ml" / "ner" / "annotation"

    reviewed_file = annotation_dir / "annotated_ner_reviewed.jsonl"
    sample_file = annotation_dir / "annotation_sample.jsonl"
    review_progress_file = annotation_dir / "review_progress.json"
    report_file = annotation_dir / "reviewed_ner_report.txt"

    # Import validator logic from validate_annotations.py
    sys.path.insert(0, str(annotation_dir))
    from validate_annotations import HumanNERValidator
    validator = HumanNERValidator(approved_labels=APPROVED_LABELS)

    # 1. Total sample records expected
    sample_count = 400
    sample_ids = set()
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    sample_ids.add(rec["id"])
        sample_count = len(sample_ids)

    if not reviewed_file.exists():
        logger.info(f"Reviewed dataset file not found at: {reviewed_file}")
        report_text = (
            "================================================================================\n"
            "HUMAN NER REVIEWED DATASET VALIDATION REPORT\n"
            "================================================================================\n"
            f"Reviewed Dataset File: {reviewed_file}\n"
            "STATUS: HUMAN REVIEW NOT COMPLETED (0 records reviewed)\n"
            f"Total Sample Narratives Queued: {sample_count}\n"
            "Remaining Records to Review: 400\n"
            "================================================================================\n"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(report_text)
        return False, report_text

    reviewed_records = []
    seen_ids = set()
    duplicate_ids = 0
    validation_errors = []

    entities_by_type = {lbl: 0 for lbl in APPROVED_LABELS}
    total_entities = 0
    records_zero_entities = 0

    invalid_spans_cnt = 0
    duplicate_spans_cnt = 0
    overlapping_spans_cnt = 0
    malformed_records_cnt = 0

    with open(reviewed_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                malformed_records_cnt += 1
                validation_errors.append(f"Line {line_no}: JSON decode error - {e}")
                continue

            rec_id = record.get("id")
            if rec_id in seen_ids:
                duplicate_ids += 1
                validation_errors.append(f"Duplicate record ID in reviewed dataset: '{rec_id}'")
            seen_ids.add(rec_id)

            is_valid, errors, stats = validator.validate_record(record)
            invalid_spans_cnt += stats["invalid_spans"]
            duplicate_spans_cnt += stats["duplicate_spans"]
            overlapping_spans_cnt += stats["overlapping_spans"]
            malformed_records_cnt += stats["malformed_records"]

            if errors:
                for err in errors:
                    validation_errors.append(f"Record '{rec_id}': {err}")

            entities = record.get("entities", [])
            if not entities:
                records_zero_entities += 1
            else:
                for ent in entities:
                    total_entities += 1
                    lbl = ent.get("label")
                    if lbl in entities_by_type:
                        entities_by_type[lbl] += 1

            reviewed_records.append(record)

    total_reviewed = len(reviewed_records)
    remaining = max(0, sample_count - total_reviewed)
    pct_complete = (total_reviewed / sample_count * 100) if sample_count > 0 else 0.0

    passed_all = (
        total_reviewed == sample_count
        and duplicate_ids == 0
        and len(validation_errors) == 0
    )

    # Save review_progress.json
    progress_data = {
        "total_records": sample_count,
        "reviewed_records": total_reviewed,
        "remaining_records": remaining,
        "annotation_percentage": round(pct_complete, 2),
        "entities_by_type": entities_by_type,
        "total_entities": total_entities,
        "records_with_zero_entities": records_zero_entities,
        "passed_validation": passed_all,
        "validation_errors": len(validation_errors),
    }

    with open(review_progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2)

    # Build report text
    report_lines = [
        "=" * 80,
        "HUMAN NER REVIEWED DATASET VALIDATION REPORT",
        "=" * 80,
        f"Reviewed Dataset File: {reviewed_file}",
        f"Total Sample Narratives Queued: {sample_count}",
        f"Total Reviewed Records: {total_reviewed} ({pct_complete:.1f}% complete)",
        f"Remaining Records: {remaining}",
        f"Unique Record IDs: {len(seen_ids)} (Duplicate IDs: {duplicate_ids})",
        f"Records with Zero Entities: {records_zero_entities}",
        "-" * 80,
        f"TOTAL FINAL ENTITY SPANS: {total_entities}",
        "FINAL ENTITY COUNTS BY TYPE:",
    ]

    for lbl, cnt in entities_by_type.items():
        report_lines.append(f"  - {lbl:<12}: {cnt}")

    report_lines.extend([
        "-" * 80,
        "STRUCTURAL VALIDATION QUALITY AUDIT:",
        f"  - Passed Full Validation: {'YES (READY FOR NER TRAINING)' if passed_all else 'NO (INCOMPLETE OR HAS ERRORS)'}",
        f"  - Invalid Spans / Offset Mismatches: {invalid_spans_cnt}",
        f"  - Duplicate Spans: {duplicate_spans_cnt}",
        f"  - Overlapping Spans: {overlapping_spans_cnt}",
        f"  - Malformed Records: {malformed_records_cnt}",
        f"  - Total Validation Errors: {len(validation_errors)}",
        "-" * 80,
        "DETAILED VALIDATION ERROR LOG:",
    ])

    if not validation_errors:
        report_lines.append("  - None. All reviewed records passed structural validation perfectly.")
    else:
        for err in validation_errors[:20]:
            report_lines.append(f"  - {err}")
        if len(validation_errors) > 20:
            report_lines.append(f"  - ... and {len(validation_errors) - 20} more errors.")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    return passed_all, report_text


if __name__ == "__main__":
    validate_reviewed_dataset()
