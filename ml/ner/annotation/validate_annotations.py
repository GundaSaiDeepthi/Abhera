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
logger = logging.getLogger("ValidateAnnotations")

APPROVED_LABELS = ["PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM", "EVIDENCE", "LAW_SEC"]


class HumanNERValidator:
    """Validates human-annotated NER records."""

    def __init__(self, approved_labels: List[str] = APPROVED_LABELS):
        self.approved_labels = set(approved_labels)

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, int]]:
        """
        Validates an individual annotated record.
        Returns (is_valid, list_of_errors, stats_dict).
        """
        errors = []
        stats = {
            "invalid_spans": 0,
            "duplicate_spans": 0,
            "overlapping_spans": 0,
            "malformed_records": 0,
        }

        if not isinstance(record, dict):
            stats["malformed_records"] += 1
            return False, ["Record is not a valid JSON object."], stats

        for req_key in ["id", "text", "entities"]:
            if req_key not in record:
                errors.append(f"Missing required key: '{req_key}'")

        if errors:
            stats["malformed_records"] += 1
            return False, errors, stats

        text = str(record["text"])
        entities = record["entities"]

        if not isinstance(entities, list):
            stats["malformed_records"] += 1
            return False, ["Field 'entities' must be a list."], stats

        spans_seen = set()
        occupied_chars = []

        for idx, ent in enumerate(entities):
            if not isinstance(ent, dict):
                errors.append(f"Entity at index {idx} is not a dictionary.")
                stats["malformed_records"] += 1
                continue

            for req_f in ["start", "end", "label"]:
                if req_f not in ent:
                    errors.append(f"Entity #{idx} missing field '{req_f}'")
                    stats["malformed_records"] += 1

            if errors:
                continue

            start = ent["start"]
            end = ent["end"]
            label = ent["label"]

            # Type checks & bounds
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"Entity #{idx}: start/end must be integers.")
                stats["invalid_spans"] += 1
                continue

            if start < 0 or end > len(text) or start >= end:
                errors.append(
                    f"Entity #{idx}: invalid bounds [{start}, {end}] for text length {len(text)}."
                )
                stats["invalid_spans"] += 1

            if label not in self.approved_labels:
                errors.append(f"Entity #{idx}: unapproved label '{label}'.")
                stats["invalid_spans"] += 1

            # Exact text slice verification
            if "text" in ent:
                actual_slice = text[start:end]
                expected_slice = ent["text"]
                if actual_slice != expected_slice:
                    errors.append(
                        f"Entity #{idx}: slice mismatch. Stored '{expected_slice}', text[{start}:{end}] is '{actual_slice}'."
                    )
                    stats["invalid_spans"] += 1

            # Duplicate span check
            span_key = (start, end, label)
            if span_key in spans_seen:
                errors.append(f"Entity #{idx}: duplicate span {span_key}.")
                stats["duplicate_spans"] += 1
            spans_seen.add(span_key)

            # Overlap check
            for c_idx in range(start, end):
                if c_idx in occupied_chars:
                    errors.append(f"Entity #{idx}: character offset {c_idx} overlaps another span.")
                    stats["overlapping_spans"] += 1
                    break
                else:
                    occupied_chars.append(c_idx)

        is_valid = len(errors) == 0
        return is_valid, errors, stats


def validate_annotated_dataset(target_file: Path = None):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    annotation_dir = project_root / "ml" / "ner" / "annotation"

    if target_file is None:
        if (annotation_dir / "annotated_ner.jsonl").exists():
            annotated_file = annotation_dir / "annotated_ner.jsonl"
        elif (annotation_dir / "annotated_ner_draft.jsonl").exists():
            annotated_file = annotation_dir / "annotated_ner_draft.jsonl"
        else:
            annotated_file = annotation_dir / "annotated_ner.jsonl"
    else:
        annotated_file = Path(target_file)
    sample_file = annotation_dir / "annotation_sample.jsonl"
    progress_file = annotation_dir / "annotation_progress.json"

    validator = HumanNERValidator()

    if not annotated_file.exists():
        logger.info(f"Annotated dataset file not found at: {annotated_file}")
        logger.info("Human annotation has not been started yet. Running validation sanity check on empty state...")
        
        # Total sample records count
        sample_count = 0
        if sample_file.exists():
            with open(sample_file, "r", encoding="utf-8") as f:
                sample_count = sum(1 for line in f if line.strip())

        print("==================================================")
        print("HUMAN NER ANNOTATION VALIDATION REPORT")
        print("==================================================")
        print(f"Annotated Dataset File: {annotated_file}")
        print("Status: ANNOTATION NOT STARTED YET (0 records annotated)")
        print(f"Total Sample Narratives Queued: {sample_count}")
        print("==================================================")
        return

    total_annotated_records = 0
    total_entities = 0
    entities_by_type = {lbl: 0 for lbl in APPROVED_LABELS}
    records_with_no_entities = 0
    total_invalid_spans = 0
    total_duplicate_spans = 0
    total_overlapping_spans = 0
    total_malformed_records = 0
    all_validation_errors = []

    with open(annotated_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total_annotated_records += 1
            except Exception as e:
                total_malformed_records += 1
                all_validation_errors.append(f"Line {line_no}: JSON decode error - {e}")
                continue

            is_valid, errors, stats = validator.validate_record(record)
            
            total_invalid_spans += stats["invalid_spans"]
            total_duplicate_spans += stats["duplicate_spans"]
            total_overlapping_spans += stats["overlapping_spans"]
            total_malformed_records += stats["malformed_records"]

            if errors:
                for err in errors:
                    all_validation_errors.append(f"Record {record.get('id', line_no)}: {err}")

            entities = record.get("entities", [])
            if not entities:
                records_with_no_entities += 1
            else:
                for ent in entities:
                    total_entities += 1
                    lbl = ent.get("label")
                    if lbl in entities_by_type:
                        entities_by_type[lbl] += 1

    sample_count = 400
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_count = sum(1 for line in f if line.strip())

    remaining_records = max(0, sample_count - total_annotated_records)
    pct_complete = (total_annotated_records / sample_count * 100) if sample_count > 0 else 0.0

    # Update progress tracking JSON
    progress_data = {
        "total_records": sample_count,
        "annotated_records": total_annotated_records,
        "remaining_records": remaining_records,
        "annotation_percentage": round(pct_complete, 2),
        "entities_by_type": entities_by_type,
        "total_entities": total_entities,
        "records_with_no_entities": records_with_no_entities,
        "validation_errors": len(all_validation_errors),
        "invalid_spans": total_invalid_spans,
        "duplicate_spans": total_duplicate_spans,
        "overlapping_spans": total_overlapping_spans,
        "malformed_records": total_malformed_records,
    }

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2)

    print("==================================================")
    print("HUMAN NER ANNOTATION VALIDATION REPORT")
    print("==================================================")
    print(f"Annotated File: {annotated_file}")
    print(f"Total Sample Narratives: {sample_count}")
    print(f"Total Annotated Records: {total_annotated_records} ({pct_complete:.1f}% complete)")
    print(f"Remaining Narratives: {remaining_records}")
    print(f"Total Entities Extracted: {total_entities}")
    print(f"Records With No Entities: {records_with_no_entities}")
    print("--------------------------------------------------")
    print("ENTITIES BY TYPE:")
    for lbl, cnt in entities_by_type.items():
        print(f"  - {lbl:<12}: {cnt}")
    print("--------------------------------------------------")
    print("QUALITY & VALIDATION AUDIT:")
    print(f"  - Invalid Spans: {total_invalid_spans}")
    print(f"  - Duplicate Spans: {total_duplicate_spans}")
    print(f"  - Overlapping Spans: {total_overlapping_spans}")
    print(f"  - Malformed Records: {total_malformed_records}")
    print(f"  - Total Validation Errors: {len(all_validation_errors)}")
    print("==================================================")


if __name__ == "__main__":
    validate_annotated_dataset()
