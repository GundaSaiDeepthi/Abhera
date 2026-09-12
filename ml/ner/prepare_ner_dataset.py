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
logger = logging.getLogger("NERPreparation")


# Placeholder schema configuration for approved entity types once annotated
APPROVED_ENTITY_SCHEMA = {
    "PERP_REL": "Perpetrator Relation (e.g., husband, boss, stranger)",
    "LOCATION": "Harassment Location (e.g., office, home, bus stand)",
    "TIME_FREQ": "Time or Frequency (e.g., every morning, daily)",
    "PLATFORM": "Digital Platform (e.g., social media, email, fake profile)",
    "EVIDENCE": "Physical Harm or Evidence (e.g., belt, slapped, locked inside)",
    "LAW_SEC": "Statutory Law Section (e.g., 354 IPC, POSH Act)",
}


def resolve_dataset_path() -> Path:
    """Finds the path to data/womens_safety_dataset.csv dynamically."""
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent.parent / "data" / "womens_safety_dataset.csv",
        Path.cwd() / "data" / "womens_safety_dataset.csv",
        Path.cwd().parent / "data" / "womens_safety_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate womens_safety_dataset.csv. Checked: {[str(c) for c in candidates]}"
    )


class NERAnnotationValidator:
    """Validation utilities for verifying future human-annotated NER dataset records."""

    def __init__(self, approved_labels: List[str]):
        self.approved_labels = set(approved_labels)

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a single annotation record.

        Checks performed:
        - Presence of required keys ('id', 'text', 'entities')
        - Text is non-empty string
        - Valid start/end offsets (0 <= start < end <= len(text))
        - Entity label belongs to approved schema
        - Exact character slice matches declared span text (if provided)
        - No duplicate spans
        - No overlapping spans
        """
        errors = []
        if not isinstance(record, dict):
            return False, ["Record must be a JSON dictionary."]

        for key in ["id", "text", "entities"]:
            if key not in record:
                errors.append(f"Missing required key: '{key}'")

        if errors:
            return False, errors

        text = record["text"]
        entities = record["entities"]

        if not isinstance(text, str) or not text:
            errors.append("Field 'text' must be a non-empty string.")

        if not isinstance(entities, list):
            errors.append("Field 'entities' must be a list of span objects.")
            return False, errors

        spans_seen = set()
        occupied_chars = []

        for idx, ent in enumerate(entities):
            if not isinstance(ent, dict):
                errors.append(f"Entity at index {idx} is not a dictionary.")
                continue

            for field in ["start", "end", "label"]:
                if field not in ent:
                    errors.append(f"Entity #{idx} missing required field: '{field}'")

            if errors:
                continue

            start = ent["start"]
            end = ent["end"]
            label = ent["label"]

            # 1. Type checks
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"Entity #{idx}: 'start' and 'end' must be integers.")
                continue

            # 2. Offset sanity
            if start < 0:
                errors.append(f"Entity #{idx}: 'start' ({start}) cannot be negative.")
            if end > len(text):
                errors.append(f"Entity #{idx}: 'end' ({end}) exceeds text length ({len(text)}).")
            if start >= end:
                errors.append(f"Entity #{idx}: 'start' ({start}) must be strictly less than 'end' ({end}).")

            # 3. Label validity
            if label not in self.approved_labels:
                errors.append(f"Entity #{idx}: label '{label}' is not in approved schema.")

            # 4. Text slice verification (if span text provided)
            if "text" in ent:
                actual_slice = text[start:end]
                expected_slice = ent["text"]
                if actual_slice != expected_slice:
                    errors.append(
                        f"Entity #{idx}: text slice mismatch. Recorded '{expected_slice}', "
                        f"actual slice text[{start}:{end}] is '{actual_slice}'."
                    )

            # 5. Duplicate check
            span_key = (start, end, label)
            if span_key in spans_seen:
                errors.append(f"Entity #{idx}: duplicate span recorded {span_key}.")
            spans_seen.add(span_key)

            # 6. Overlap check
            for char_idx in range(start, end):
                if char_idx in occupied_chars:
                    errors.append(f"Entity #{idx}: character offset {char_idx} overlaps with another span.")
                else:
                    occupied_chars.append(char_idx)

        return (len(errors) == 0), errors


def convert_spans_to_bio_workflow_demo(text: str, entities: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Demonstrates the conversion workflow from character spans to token-level BIO tags.
    This workflow will be utilized once human annotations are available.
    """
    tokens = text.split()
    bio_tags = []
    
    # Simple whitespace tokenization mapping for workflow demonstration
    char_to_bio = {}
    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        for c in range(start, end):
            if c == start:
                char_to_bio[c] = f"B-{label}"
            else:
                char_to_bio[c] = f"I-{label}"

    curr_offset = 0
    for token in tokens:
        start_c = text.find(token, curr_offset)
        end_c = start_c + len(token)
        curr_offset = end_c

        # Tag assignment
        token_bio = "O"
        for c in range(start_c, end_c):
            if c in char_to_bio:
                token_bio = char_to_bio[c]
                break
        bio_tags.append((token, token_bio))

    return bio_tags


def run_ner_preparation():
    project_root = Path(__file__).resolve().parent.parent.parent
    ner_dir = project_root / "ml" / "ner"
    ner_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_dataset_path()
    logger.info(f"Loading incident dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    total_records = len(df)
    columns_list = list(df.columns)

    logger.info(f"Verifying NER annotation presence across {total_records} records...")

    # Verification: check if any explicit span/BIO columns exist
    ner_span_cols = [c for c in columns_list if any(kw in c.lower() for kw in ["ner", "bio", "span", "entity_offsets"])]
    has_ner_data = len(ner_span_cols) > 0

    if not has_ner_data:
        logger.warning("VERIFICATION COMPLETE: Zero explicit NER token/span annotations found in dataset.")
        logger.warning("Strict Zero-Fabrication Enforcement: BIO tags WILL NOT be artificially generated.")
    else:
        logger.info(f"Detected explicit NER columns: {ner_span_cols}")

    # Validator initialization
    validator = NERAnnotationValidator(approved_labels=list(APPROVED_ENTITY_SCHEMA.keys()))

    # Build Preparation Report
    report_lines = [
        "=" * 80,
        "NAMED ENTITY RECOGNITION (NER) DATASET PREPARATION REPORT (PART 11 STEP 2)",
        "=" * 80,
        f"1. CURRENT DATASET STATUS & VERIFICATION:",
        f"   - Dataset File: {csv_path}",
        f"   - Total Incident Narratives: {total_records}",
        f"   - Columns Present ({len(columns_list)}): {', '.join(columns_list)}",
        f"   - Explicit Span/BIO Annotations Present: {'YES' if has_ner_data else 'NO'}",
        "",
        "2. REASONING FOR ZERO-FABRICATION POLICY:",
        "   - The dataset contains narrative texts and multi-label document classifications.",
        "   - It does NOT contain token-level BIO tags or ground-truth character offset spans.",
        "   - Synthesizing BIO labels via regex rules or LLM guesswork is strictly prohibited,",
        "     as it introduces unvalidated label noise and violates zero-hallucination standards.",
        "",
        "3. REQUIRED ANNOTATION SCHEMA (PLACEHOLDER SPECIFICATION):",
    ]

    for label_code, desc in APPROVED_ENTITY_SCHEMA.items():
        report_lines.append(f"   - {label_code:<12}: {desc}")

    report_lines.extend([
        "",
        "4. VALIDATION CHECKS IMPLEMENTED (IN prepare_ner_dataset.py):",
        "   - Valid start/end offsets (0 <= start < end <= text_length)",
        "   - Text slice matching declared entity string",
        "   - Approved entity label schema adherence",
        "   - Detection of duplicate entity spans",
        "   - Detection of overlapping entity character spans",
        "   - Rejection of malformed records & empty fields",
        "",
        "5. PLANNED END-TO-END CONVERSION WORKFLOW FOR FUTURE HUMAN ANNOTATIONS:",
        "   [Human Annotators (.jsonl)] -> [NERAnnotationValidator] -> [Span-to-BIO Alignment]",
        "   -> [BERT WordPiece Tokenizer Alignment] -> [70/15/15 Train/Val/Test Split]",
        "   -> [Supervised Token Classification Model Training (bert-base-uncased-NER)]",
        "",
        "6. FEASIBILITY CONCLUSION & PRE-TRAINING REQUIREMENTS:",
        "   - Can NER Training Begin Now? NO.",
        "   - Required Action Before Training: A human-annotated dataset adhering to the schema",
        "     and validated by NERAnnotationValidator must be created using ner_annotation_template.jsonl.",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save preparation report
    report_file_path = ner_dir / "ner_preparation_report.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved NER preparation report to: {report_file_path}")

    return {
        "dataset_path": str(csv_path),
        "total_records": total_records,
        "has_ner_annotations": has_ner_data,
        "approved_schema": APPROVED_ENTITY_SCHEMA,
        "report_path": str(report_file_path),
    }


if __name__ == "__main__":
    run_ner_preparation()
