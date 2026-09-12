import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BuildProvisionalNER")

APPROVED_LABELS = {"PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM", "EVIDENCE", "LAW_SEC"}


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


def build_provisional_dataset():
    project_root = get_project_root()
    annotation_dir = project_root / "ml" / "ner" / "annotation"

    sample_jsonl_path = annotation_dir / "annotation_sample.jsonl"
    auto_accept_jsonl_path = annotation_dir / "ner_second_pass_auto_accept.jsonl"
    human_review_jsonl_path = annotation_dir / "ner_second_pass_human_review.jsonl"
    auto_reject_jsonl_path = annotation_dir / "ner_second_pass_auto_reject.jsonl"

    provisional_dataset_path = annotation_dir / "provisional_ner_dataset.jsonl"
    provisional_review_path = annotation_dir / "provisional_ner_human_review_remaining.jsonl"
    report_file_path = annotation_dir / "provisional_ner_report.txt"

    if not sample_jsonl_path.exists():
        logger.error(f"Sample JSONL file missing: {sample_jsonl_path}")
        sys.exit(1)

    if not auto_accept_jsonl_path.exists():
        logger.error(f"Auto-accept candidates JSONL missing: {auto_accept_jsonl_path}")
        sys.exit(1)

    # 1. Load all 400 sample records to preserve complete narrative set and stable IDs
    sample_records = []
    sample_map = {}
    with open(sample_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                sample_records.append(rec)
                sample_map[rec["id"]] = rec["text"]

    total_source_records = len(sample_records)
    logger.info(f"Loaded {total_source_records} source records from {sample_jsonl_path}.")

    # 2. Load second-pass AUTO_ACCEPT candidates
    auto_accept_entries = []
    with open(auto_accept_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                auto_accept_entries.append(json.loads(line))

    logger.info(f"Loaded {len(auto_accept_entries)} AUTO_ACCEPT candidates.")

    # Load HUMAN_REVIEW and AUTO_REJECT candidates for auditing metrics
    human_review_entries = []
    if human_review_jsonl_path.exists():
        with open(human_review_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    human_review_entries.append(json.loads(line))

    auto_reject_entries = []
    if auto_reject_jsonl_path.exists():
        with open(auto_reject_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    auto_reject_entries.append(json.loads(line))

    # 3. Comprehensive Integrity Validation on AUTO_ACCEPT candidates
    validation_failures = 0
    duplicate_spans_count = 0
    overlapping_spans_count = 0
    malformed_records_count = 0
    validation_error_messages = []

    # Map candidate entities by record ID
    record_accepted_entities: Dict[str, List[Dict[str, Any]]] = {rec["id"]: [] for rec in sample_records}
    entity_counts_by_label = {lbl: 0 for lbl in APPROVED_LABELS}

    for idx, entry in enumerate(auto_accept_entries, 1):
        rec_id = entry.get("record_id")
        cand_text = entry.get("candidate_text")
        start = entry.get("start")
        end = entry.get("end")
        label = entry.get("proposed_label")
        narrative_text = entry.get("narrative_text") or sample_map.get(rec_id, "")

        # Check record ID validity
        if not rec_id or rec_id not in sample_map:
            validation_failures += 1
            validation_error_messages.append(f"Entry #{idx}: Record ID '{rec_id}' not found in sample records.")
            continue

        # Check narrative presence
        if not narrative_text:
            validation_failures += 1
            validation_error_messages.append(f"Entry #{idx}: Empty narrative text for record '{rec_id}'.")
            continue

        # Check offsets sanity
        if start is None or end is None or not isinstance(start, int) or not isinstance(end, int):
            validation_failures += 1
            validation_error_messages.append(f"Entry #{idx} (Record '{rec_id}'): Start/end offsets must be integers.")
            continue

        if start < 0 or end > len(narrative_text) or start >= end:
            validation_failures += 1
            validation_error_messages.append(f"Entry #{idx} (Record '{rec_id}'): Invalid bounds [{start}:{end}] for text length {len(narrative_text)}.")
            continue

        # Exact text slice match
        actual_slice = narrative_text[start:end]
        if actual_slice != cand_text:
            validation_failures += 1
            validation_error_messages.append(
                f"Entry #{idx} (Record '{rec_id}'): Slice mismatch. Candidate '{cand_text}', narrative slice text[{start}:{end}] is '{actual_slice}'."
            )
            continue

        # Label validity
        if label not in APPROVED_LABELS:
            validation_failures += 1
            validation_error_messages.append(f"Entry #{idx} (Record '{rec_id}'): Unapproved label '{label}'.")
            continue

        # Check duplicates & overlaps within target record
        existing_ents = record_accepted_entities[rec_id]
        span_key = (start, end, label)

        is_duplicate = any((e["start"], e["end"], e["label"]) == span_key for e in existing_ents)
        if is_duplicate:
            duplicate_spans_count += 1
            validation_failures += 1
            validation_error_messages.append(f"Record '{rec_id}': Duplicate accepted span {span_key}.")
            continue

        is_overlap = False
        for e in existing_ents:
            if not (end <= e["start"] or start >= e["end"]):
                is_overlap = True
                overlapping_spans_count += 1
                validation_failures += 1
                validation_error_messages.append(
                    f"Record '{rec_id}': Span [{start}:{end}] overlaps existing span [{e['start']}:{e['end']}]."
                )
                break

        if is_overlap:
            continue

        # Passed all integrity checks! Add to accepted list
        accepted_span = {
            "start": start,
            "end": end,
            "label": label,
            "text": cand_text,
        }
        record_accepted_entities[rec_id].append(accepted_span)
        if label in entity_counts_by_label:
            entity_counts_by_label[label] += 1

    # 4. Construct Provisional Dataset JSONL Records
    provisional_records = []
    records_with_accepted_entities = 0
    records_without_accepted_entities = 0
    total_accepted_entities = 0

    for rec in sample_records:
        rec_id = rec["id"]
        text = rec["text"]
        ents = record_accepted_entities.get(rec_id, [])

        # Sort entities by start offset
        ents.sort(key=lambda x: x["start"])

        if ents:
            records_with_accepted_entities += 1
            total_accepted_entities += len(ents)
        else:
            records_without_accepted_entities += 1

        provisional_records.append({
            "id": rec_id,
            "text": text,
            "entities": ents,
        })

    # 5. Write Output JSONL Files
    with open(provisional_dataset_path, "w", encoding="utf-8") as f:
        for rec in provisional_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved provisional dataset ({len(provisional_records)} records) to: {provisional_dataset_path}")

    # Write remaining human review candidates file
    with open(provisional_review_path, "w", encoding="utf-8") as f:
        for entry in human_review_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved remaining human review candidates ({len(human_review_entries)} entries) to: {provisional_review_path}")

    # 6. Sample Accepted Annotations for Report
    sample_accepted_examples = []
    for rec in provisional_records:
        for e in rec["entities"]:
            sample_accepted_examples.append(
                f"Record '{rec['id']}': '{e['text']}' -> {e['label']} [{e['start']}:{e['end']}]"
            )
            if len(sample_accepted_examples) >= 8:
                break
        if len(sample_accepted_examples) >= 8:
            break

    # 7. Generate Summary Report
    report_lines = [
        "=" * 80,
        "PROVISIONAL MACHINE-VALIDATED NER DATASET REPORT (PART 11 STEP 9)",
        "=" * 80,
        "DATASET STATUS NOTICE:",
        "PROVISIONAL MACHINE-VALIDATED NER DATASET — NOT HUMAN-GROUNDED-TRUTH",
        "Under strict zero-fabrication guidelines, machine-accepted annotations are kept",
        "distinct from human-adjudicated ground truth and must be human-reviewed before final training.",
        "-" * 80,
        "1. DATASET SUMMARY & METRICS:",
        f"   - Number of Source Sample Records: {total_source_records}",
        f"   - Number of Accepted Candidate Entities: {total_accepted_entities}",
        f"   - Records Containing Accepted Entities: {records_with_accepted_entities} ({records_with_accepted_entities/total_source_records*100:.1f}%)",
        f"   - Records Without Accepted Entities: {records_without_accepted_entities} ({records_without_accepted_entities/total_source_records*100:.1f}%)",
        f"   - Remaining Human-Review Candidates: {len(human_review_entries)}",
        f"   - Rejected Candidates Count: {len(auto_reject_entries)}",
        "",
        "2. ACCEPTED ENTITY COUNTS BY LABEL:",
    ]

    for lbl, cnt in entity_counts_by_label.items():
        report_lines.append(f"   - {lbl:<10}: {cnt}")

    report_lines.extend([
        "",
        "3. INTEGRITY & VALIDATION AUDIT:",
        f"   - Validation Failures Count: {validation_failures}",
        f"   - Duplicate Spans Count: {duplicate_spans_count}",
        f"   - Overlapping Spans Count: {overlapping_spans_count}",
        f"   - Malformed Records Count: {malformed_records_count}",
        f"   - Passed Full Integrity Audit: {'YES' if validation_failures == 0 else 'NO'}",
        "",
        "4. EXAMPLES OF ACCEPTED ANNOTATIONS:",
    ])

    for ex in sample_accepted_examples:
        report_lines.append(f"   - {ex}")

    report_lines.extend([
        "",
        "5. ARTIFACT LOCATIONS:",
        f"   - Provisional Dataset: {provisional_dataset_path}",
        f"   - Remaining Human Review Queue: {provisional_review_path}",
        f"   - Preparation Report: {report_file_path}",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved provisional report to: {report_file_path}")

    return {
        "total_source_records": total_source_records,
        "total_accepted_entities": total_accepted_entities,
        "records_with_accepted_entities": records_with_accepted_entities,
        "records_without_accepted_entities": records_without_accepted_entities,
        "entity_counts_by_label": entity_counts_by_label,
        "human_review_candidates": len(human_review_entries),
        "rejected_candidates": len(auto_reject_entries),
        "validation_failures": validation_failures,
        "duplicate_count": duplicate_spans_count,
        "overlap_count": overlapping_spans_count,
        "malformed_records": malformed_records_count,
        "provisional_dataset_path": str(provisional_dataset_path),
    }


if __name__ == "__main__":
    build_provisional_dataset()
