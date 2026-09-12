import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ImportNERDraft")

APPROVED_LABELS = {"PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM", "EVIDENCE", "LAW_SEC"}


def parse_annotation_string(annotation_str: str, text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parses an annotation string format:
    'phrase1 -> LABEL1 [start1,end1] | phrase2 -> LABEL2 [start2,end2]'
    """
    entities = []
    errors = []

    if not isinstance(annotation_str, str) or not annotation_str.strip() or annotation_str.strip().upper() == "NONE":
        return entities, errors

    parts = [p.strip() for p in annotation_str.split("|") if p.strip()]

    for part in parts:
        # Pattern matching: phrase -> LABEL [start,end] OR phrase -> LABEL
        match_with_offsets = re.match(r"^(.*?)\s*->\s*([A-Z_]+)\s*\[(\d+)\s*,\s*(\d+)\]$", part)
        match_no_offsets = re.match(r"^(.*?)\s*->\s*([A-Z_]+)$", part)

        if match_with_offsets:
            phrase = match_with_offsets.group(1).strip()
            label = match_with_offsets.group(2).strip()
            start = int(match_with_offsets.group(3))
            end = int(match_with_offsets.group(4))

            # Verification of provided offsets
            if start < 0 or end > len(text) or start >= end:
                errors.append(f"Invalid offsets [{start}, {end}] for phrase '{phrase}' in text length {len(text)}.")
                continue

            actual_slice = text[start:end]
            if actual_slice != phrase:
                # Try finding exact phrase in text
                if phrase in text:
                    found_start = text.find(phrase)
                    found_end = found_start + len(phrase)
                    errors.append(
                        f"Offset mismatch for '{phrase}': CSV declared [{start},{end}] ('{actual_slice}'), "
                        f"re-aligned to [{found_start},{found_end}]."
                    )
                    start, end = found_start, found_end
                else:
                    errors.append(f"Phrase '{phrase}' not found in narrative text: '{text}'.")
                    continue

        elif match_no_offsets:
            phrase = match_no_offsets.group(1).strip()
            label = match_no_offsets.group(2).strip()

            if phrase in text:
                start = text.find(phrase)
                end = start + len(phrase)
            else:
                errors.append(f"Phrase '{phrase}' not found in narrative text.")
                continue
        else:
            errors.append(f"Malformed annotation entry: '{part}'")
            continue

        if label not in APPROVED_LABELS:
            errors.append(f"Unsupported label '{label}' for phrase '{phrase}'. Approved labels: {sorted(list(APPROVED_LABELS))}.")

        entities.append({
            "start": start,
            "end": end,
            "label": label,
            "text": phrase,
        })

    return entities, errors


def run_draft_import():
    project_root = Path(__file__).resolve().parent.parent.parent
    ner_dir = project_root / "ml" / "ner"
    annotation_dir = ner_dir / "annotation"

    csv_path = annotation_dir / "annotated_ner_draft_review.csv"
    sample_jsonl_path = annotation_dir / "annotation_sample.jsonl"
    draft_jsonl_path = annotation_dir / "annotated_ner_draft.jsonl"
    report_path = annotation_dir / "draft_annotation_report.txt"

    if not csv_path.exists():
        logger.error(f"Draft CSV file not found at: {csv_path}")
        sys.exit(1)

    if not sample_jsonl_path.exists():
        logger.error(f"Sample JSONL file not found at: {sample_jsonl_path}")
        sys.exit(1)

    # 1. Load sample JSONL to map ground-truth sample IDs and narrative texts
    sample_map = {}
    with open(sample_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                sample_map[rec["id"]] = rec["text"]

    logger.info(f"Loaded {len(sample_map)} sample narrative records from {sample_jsonl_path}.")

    # 2. Inspect CSV structure
    df = pd.read_csv(csv_path)
    total_csv_rows = len(df)
    unique_csv_ids = df["id"].nunique()
    duplicate_csv_ids = total_csv_rows - unique_csv_ids

    logger.info(f"CSV Total Rows: {total_csv_rows} | Unique IDs: {unique_csv_ids} | Duplicate IDs: {duplicate_csv_ids}")

    # Track metrics
    imported_records_count = 0
    total_proposed_entities = 0
    entities_by_type = {lbl: 0 for lbl in APPROVED_LABELS}
    records_zero_entities = 0
    valid_spans_count = 0
    invalid_spans_count = 0
    overlapping_spans_count = 0
    duplicate_spans_count = 0

    parsing_errors = []
    draft_jsonl_records = []

    for idx, row in df.iterrows():
        rec_id = str(row["id"]).strip()
        csv_text = str(row["text"]).strip() if pd.notnull(row["text"]) else ""
        annotation_raw = str(row["annotations"]).strip() if pd.notnull(row["annotations"]) else ""

        # Match against sample map
        if rec_id not in sample_map:
            parsing_errors.append(f"CSV Row #{idx+1}: ID '{rec_id}' not found in annotation_sample.jsonl.")
            narrative_text = csv_text
        else:
            narrative_text = sample_map[rec_id]
            if csv_text and csv_text != narrative_text:
                parsing_errors.append(f"Record '{rec_id}': CSV text mismatch with annotation_sample.jsonl narrative.")

        entities, errs = parse_annotation_string(annotation_raw, narrative_text)
        if errs:
            for e in errs:
                parsing_errors.append(f"Record '{rec_id}': {e}")
                invalid_spans_count += 1

        # Check for duplicates & overlaps within record
        seen_spans = set()
        occupied_chars = []

        clean_entities = []
        for ent in entities:
            lbl = ent["label"]
            start, end = ent["start"], ent["end"]

            # Count per type
            if lbl in entities_by_type:
                entities_by_type[lbl] += 1
            total_proposed_entities += 1

            # Duplicate check
            span_key = (start, end, lbl)
            if span_key in seen_spans:
                duplicate_spans_count += 1
                parsing_errors.append(f"Record '{rec_id}': Duplicate entity span {span_key}.")
            else:
                seen_spans.add(span_key)

            # Overlap check
            is_overlap = False
            for c in range(start, end):
                if c in occupied_chars:
                    is_overlap = True
                    overlapping_spans_count += 1
                    parsing_errors.append(f"Record '{rec_id}': Entity '{ent['text']}' [{start}:{end}] overlaps another span.")
                    break
                else:
                    occupied_chars.append(c)

            if not is_overlap and span_key in seen_spans:
                valid_spans_count += 1

            clean_entities.append(ent)

        if len(clean_entities) == 0:
            records_zero_entities += 1

        draft_jsonl_records.append({
            "id": rec_id,
            "text": narrative_text,
            "entities": clean_entities,
        })
        imported_records_count += 1

    # 3. Write draft_jsonl file
    with open(draft_jsonl_path, "w", encoding="utf-8") as f:
        for rec in draft_jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved reviewable draft JSONL to: {draft_jsonl_path}")

    # 4. Run existing validation logic from validate_annotations.py
    sys.path.insert(0, str(annotation_dir))
    from validate_annotations import HumanNERValidator
    validator = HumanNERValidator(approved_labels=list(APPROVED_LABELS))

    val_errors = []
    for rec in draft_jsonl_records:
        is_val, errs, stats = validator.validate_record(rec)
        if errs:
            for e in errs:
                val_errors.append(f"Validation Error [Record {rec['id']}]: {e}")

    passed_validation = (len(parsing_errors) == 0) and (len(val_errors) == 0)

    # 5. Generate draft_annotation_report.txt
    report_lines = [
        "=" * 80,
        "DRAFT NER ANNOTATION IMPORT & AUDIT REVIEW REPORT (PART 11 STEP 4)",
        "=" * 80,
        "NOTICE: This document reports on DRAFT proposed annotations imported from",
        "annotated_ner_draft_review.csv. These are DRAFT proposed annotations requiring",
        "thorough human review before any future supervised NER training can take place.",
        "Under strict zero-fabrication rules, this draft is NOT yet ground-truth data.",
        "-" * 80,
        "1. DRAFT CSV IMPORT & OVERVIEW:",
        f"   - Draft CSV File: {csv_path}",
        f"   - Target Sample JSONL: {sample_jsonl_path}",
        f"   - Draft JSONL Created: {draft_jsonl_path}",
        f"   - Total CSV Records Imported: {imported_records_count}",
        f"   - Unique Record IDs: {unique_csv_ids}",
        f"   - Duplicate Record IDs: {duplicate_csv_ids}",
        f"   - Records with Zero Proposed Entities: {records_zero_entities}",
        "",
        "2. PROPOSED ENTITY SPANS & TYPE BREAKDOWN:",
        f"   - Total Proposed Entities: {total_proposed_entities}",
        f"   - Valid Spans: {valid_spans_count}",
        f"   - Invalid / Offset Errors: {invalid_spans_count}",
        f"   - Overlapping Spans: {overlapping_spans_count}",
        f"   - Duplicate Spans: {duplicate_spans_count}",
        "   - Proposed Entity Counts by Type:",
    ]

    for lbl, cnt in entities_by_type.items():
        report_lines.append(f"     * {lbl:<10}: {cnt}")

    report_lines.extend([
        "",
        "3. STRUCTURAL VALIDATION AUDIT RESULTS:",
        f"   - Passed Full Structural Validation: {'YES' if passed_validation else 'NO'}",
        f"   - Total Parsing / Mismatch Errors: {len(parsing_errors)}",
        f"   - Total Validator Rule Violations: {len(val_errors)}",
        "",
        "4. DETAILED ERROR LOG (UNSILENCED):",
    ])

    if not parsing_errors and not val_errors:
        report_lines.append("   - None. All records matched narrative texts, character offsets, and valid schemas perfectly.")
    else:
        for err in (parsing_errors + val_errors)[:30]:  # Show top 30
            report_lines.append(f"   - {err}")
        if len(parsing_errors + val_errors) > 30:
            report_lines.append(f"   - ... and {len(parsing_errors + val_errors) - 30} more errors.")

    report_lines.extend([
        "",
        "5. NEXT STEPS & HUMAN REVIEW REQUIREMENT:",
        "   - Human domain experts must review annotated_ner_draft.jsonl using annotation_app.py",
        "     to verify, edit, or confirm proposed spans before saving to final annotated_ner.jsonl.",
        "   - NO NER model training or evaluation may occur until human review is finalized.",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Draft annotation report saved successfully to: {report_path}")

    return {
        "imported_records": imported_records_count,
        "proposed_entities": total_proposed_entities,
        "entities_by_type": entities_by_type,
        "valid_spans": valid_spans_count,
        "invalid_spans": invalid_spans_count,
        "overlapping_spans": overlapping_spans_count,
        "duplicate_spans": duplicate_spans_count,
        "draft_jsonl_path": str(draft_jsonl_path),
        "passed_validation": passed_validation,
    }


if __name__ == "__main__":
    run_draft_import()
