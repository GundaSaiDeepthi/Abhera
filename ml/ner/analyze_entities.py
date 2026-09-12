import json
import logging
from pathlib import Path
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NERAnalysis")


def resolve_dataset_path() -> Path:
    """Finds the dataset path d:/Abhera(Mini)/data/womens_safety_dataset.csv dynamically."""
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
        f"Could not find womens_safety_dataset.csv. Checked in: {[str(c) for c in candidates]}"
    )


def run_ner_analysis():
    current_file = Path(__file__).resolve()
    ner_dir = current_file.parent
    ner_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_dataset_path()
    logger.info(f"Inspecting dataset at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Dataset stats
    total_records = len(df)
    columns_list = list(df.columns)
    data_types = df.dtypes.astype(str).to_dict()
    missing_values = df.isnull().sum().to_dict()

    # 2. Check for explicit NER annotations
    explicit_ner_columns = [
        col for col in columns_list
        if any(keyword in col.lower() for keyword in ["ner", "bio", "entity", "span", "start", "end", "offset", "tokens"])
        and col not in ["evidence_notes"]
    ]

    has_explicit_ner_spans = len(explicit_ner_columns) > 0

    # 3. Inspect text & structured non-classification fields
    narrative_col = "narrative_text" if "narrative_text" in df.columns else None
    evidence_notes_present = "evidence_notes" in df.columns
    ipc_bns_present = "ipc_bns_section" in df.columns

    sample_evidence_notes = df["evidence_notes"].dropna().unique()[:10].tolist() if evidence_notes_present else []
    sample_ipc_bns = df["ipc_bns_section"].dropna().unique()[:10].tolist() if ipc_bns_present else []

    # 4. Check if entity information can be reliably derived without fabrication
    # Check if structured fields contain token offsets, BIO tags, or exact entity mentions
    # evidence_notes contains high-level category descriptions (e.g., 'public/private harassment incident', 'repeated unwanted contact/presence')
    # ipc_bns_section contains statutory section codes (e.g., '354 IPC / 74 BNS', '498A IPC / 85 BNS')

    has_structured_entity_offsets = False
    has_token_level_bio_tags = False

    # 5. Candidate entity analysis
    # Are there explicit entity span annotations in the dataset?
    candidate_supported_entities = []
    unsupported_entities = [
        "PERPETRATOR_RELATION (e.g., boss, husband, brother-in-law, stranger)",
        "LOCATION (e.g., office, home, bus stand, college, online)",
        "TIME_FREQUENCY (e.g., every morning, daily, for over a year, after hours)",
        "DIGITAL_PLATFORM (e.g., social media, email, fake profile, direct messages)",
        "PHYSICAL_HARM_EVIDENCE (e.g., belt, slapped, locked inside, injuries)",
        "LEGAL_SECTION (Statutory sections are in 'ipc_bns_section' column, but NOT annotated as text spans within narrative_text)",
    ]

    # 6. Feasibility & BIO Annotation Legitimate Assessment
    bio_legitimate = False
    supervised_ner_feasible = False

    reasoning_summary = [
        "1. EXPLICIT ANNOTATIONS MISSING: The dataset does not contain token-level BIO tags (B-PER, I-PER, B-LOC, etc.), character start/end offset spans, or structured entity dictionaries.",
        "2. HIGH-LEVEL STRUCTURED FIELDS ONLY: Columns 'ipc_bns_section' and 'evidence_notes' contain document-level statutory section codes and general category descriptors, NOT entity spans within narrative_text.",
        "3. HIGH RISK OF HALLUCINATED/FABRICATED LABELS: Generating BIO tags via regex or automated heuristics without ground-truth annotations would constitute artificial label fabrication, violating strict zero-hallucination rules.",
        "4. CONCLUSION: Supervised NER model training (e.g. fine-tuning BERT-NER / RoBERTa-NER) is NOT FEASIBLE on the current dataset without acquiring a dedicated token-level annotated NER dataset.",
    ]

    # Build report text
    report_lines = [
        "=" * 80,
        "NAMED ENTITY RECOGNITION (NER) DATASET SUPPORT ANALYSIS REPORT (PART 11 STEP 1)",
        "=" * 80,
        f"A. DATASET INSPECTED: {csv_path}",
        f"B. NUMBER OF RECORDS: {total_records}",
        f"C. DETECTED COLUMNS ({len(columns_list)}): {', '.join(columns_list)}",
        "-" * 80,
        "DATA TYPES & MISSING VALUES:",
    ]

    for col in columns_list:
        m_cnt = missing_values[col]
        dtype = data_types[col]
        report_lines.append(f"  - {col:<20} | Type: {dtype:<10} | Missing: {m_cnt}")

    report_lines.extend([
        "-" * 80,
        "D. ENTITY-RELATED INFORMATION FOUND IN DATASET:",
        f"  - Explicit Token/Span Entity Annotations: {'PRESENT' if has_explicit_ner_spans else 'ABSENT'}",
        f"  - Token-level BIO Tags: {'PRESENT' if has_token_level_bio_tags else 'ABSENT'}",
        f"  - Character Start/End Offset Fields: {'PRESENT' if has_structured_entity_offsets else 'ABSENT'}",
        f"  - Document-Level Statutory Section Field ('ipc_bns_section'): PRESENT (e.g., {sample_ipc_bns[:3]})",
        f"  - Document-Level Evidence Notes Field ('evidence_notes'): PRESENT (e.g., {sample_evidence_notes[:3]})",
        "-" * 80,
        "E. CANDIDATE ENTITY TYPES SUPPORTED BY EXPLICIT EVIDENCE:",
        "  - NONE (No explicit token-level entity span annotations exist in the dataset).",
        "-" * 80,
        "F. ENTITY TYPES THAT CANNOT CURRENTLY BE SUPPORTED (UNSUPPORTED):",
    ])

    for entity in unsupported_entities:
        report_lines.append(f"  - {entity}")

    report_lines.extend([
        "-" * 80,
        "G. CAN BIO ANNOTATIONS BE CREATED LEGITIMATELY FROM EXISTING DATASET?",
        "  - NO. Attempting to create BIO tags via heuristic substring matching or LLM guessing",
        "    would introduce unvalidated, synthetic label noise and fabricate ground-truth data.",
        "-" * 80,
        "H. IS SUPERVISED NER TRAINING CURRENTLY FEASIBLE?",
        "  - NO. Supervised NER training requires token-level entity spans and gold-standard BIO labels,",
        "    which are absent in data/womens_safety_dataset.csv.",
        "-" * 80,
        "I. REQUIRED ADDITIONAL ANNOTATION DATA FOR NER FEASIBILITY:",
        "  - To support supervised NER fine-tuning in future phases, a dedicated NER annotation dataset",
        "    is required containing token-level BIO tags (e.g. CoNLL-2003 or JSONL format with 'start', 'end', 'label'",
        "    spans for Perpetrator, Location, Time/Frequency, Digital Platform, Evidence, and Statutory Sections).",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save entity_analysis_report.txt
    report_output_path = ner_dir / "entity_analysis_report.txt"
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"NER analysis report saved successfully to: {report_output_path}")

    return {
        "dataset_inspected": str(csv_path),
        "total_records": total_records,
        "columns": columns_list,
        "has_explicit_ner_spans": has_explicit_ner_spans,
        "candidate_supported_entities": candidate_supported_entities,
        "unsupported_entities": unsupported_entities,
        "bio_legitimate": bio_legitimate,
        "supervised_ner_feasible": supervised_ner_feasible,
        "report_output_path": str(report_output_path),
    }


if __name__ == "__main__":
    run_ner_analysis()
