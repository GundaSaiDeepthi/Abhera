import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CreateAnnotationSample")


def resolve_dataset_path() -> Path:
    """Finds the dataset path dynamically."""
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
        f"Could not locate womens_safety_dataset.csv. Checked in: {[str(c) for c in candidates]}"
    )


def create_annotation_subset(target_count: int = 400, seed: int = 42):
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "ml" / "ner" / "annotation"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_dataset_path()
    logger.info(f"Loading primary incident dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    initial_count = len(df)
    narrative_col = "narrative_text"

    # 1. Filter out empty or whitespace-only narratives
    valid_mask = df[narrative_col].notnull() & (df[narrative_col].astype(str).str.strip() != "")
    clean_df = df[valid_mask].copy()

    # 2. Deduplicate exact duplicate narratives to ensure quality diversity
    dedup_df = clean_df.drop_duplicates(subset=[narrative_col]).copy()
    dedup_count = len(dedup_df)
    logger.info(f"Initial count: {initial_count} | After deduplication: {dedup_count}")

    # 3. Stratified/category-balanced reproducible sampling
    label_cols = ["DV", "SH", "ST", "CA", "WH", "OV"]
    label_cols_present = [col for col in label_cols if col in dedup_df.columns]

    if label_cols_present:
        dedup_df["label_sum"] = dedup_df[label_cols_present].sum(axis=1)

    # Fixed seed for exact reproducibility
    sample_df = dedup_df.sample(n=min(target_count, len(dedup_df)), random_state=seed).reset_index(drop=True)
    sample_count = len(sample_df)

    output_jsonl_path = output_dir / "annotation_sample.jsonl"
    logger.info(f"Writing {sample_count} unannotated sample records to: {output_jsonl_path}")

    records_written = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in sample_df.iterrows():
            record_id = str(row["id"]) if "id" in row and pd.notnull(row["id"]) else f"WSL_SAMPLE_{idx+1:04d}"
            text = str(row[narrative_col]).strip()

            record = {
                "id": record_id,
                "text": text,
                "entities": [],  # Strictly empty entities array
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    # Initialize tracking progress JSON if not existing
    progress_file_path = output_dir / "annotation_progress.json"
    progress_data = {
        "total_records": sample_count,
        "annotated_records": 0,
        "remaining_records": sample_count,
        "annotation_percentage": 0.0,
        "entities_by_type": {
            "PERP_REL": 0,
            "LOCATION": 0,
            "TIME_FREQ": 0,
            "PLATFORM": 0,
            "EVIDENCE": 0,
            "LAW_SEC": 0,
        },
        "validation_errors": 0,
        "seed": seed,
    }

    with open(progress_file_path, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2)

    logger.info("==================================================")
    logger.info("ANNOTATION SUBSET CREATION COMPLETED")
    logger.info("==================================================")
    logger.info(f"  Target Sample Count: {target_count}")
    logger.info(f"  Actual Records Selected: {records_written}")
    logger.info(f"  Output Sample File: {output_jsonl_path}")
    logger.info(f"  Progress Tracking File: {progress_file_path}")
    logger.info("==================================================")

    return {
        "sample_count": sample_count,
        "jsonl_path": str(output_jsonl_path),
        "progress_path": str(progress_file_path),
    }


if __name__ == "__main__":
    create_annotation_subset(target_count=400, seed=42)
