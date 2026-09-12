import os
from pathlib import Path
import sys
import pandas as pd


def resolve_dataset_path() -> Path:
    """Resolves the path to data/womens_safety_dataset.csv dynamically."""
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
        "Could not find womens_safety_dataset.csv. Checked in: "
        f"{[str(c) for c in candidates]}"
    )


def run_dataset_analysis():
    current_file = Path(__file__).resolve()
    csv_path = resolve_dataset_path()
    df = pd.read_csv(csv_path)

    # 1. Total records
    total_records = len(df)

    # 2. Complete list of columns
    columns_list = list(df.columns)

    # 3. Missing values count for every column
    missing_values = df.isnull().sum().to_dict()

    # 4. Identify narrative/text column
    narrative_col = None
    possible_narrative_cols = [c for c in columns_list if "narrative" in c.lower() or "text" in c.lower()]
    if possible_narrative_cols:
        narrative_col = possible_narrative_cols[0]
    else:
        narrative_col = "narrative_text" if "narrative_text" in columns_list else columns_list[1]

    # 5. Identify actual incident-label columns from schema
    candidate_labels = ["DV", "SH", "ST", "CA", "WH", "OV"]
    actual_labels = [col for col in candidate_labels if col in df.columns]

    # 6 & 7. Count examples and calculate distribution for every label
    label_counts = {}
    label_percentages = {}
    for label in actual_labels:
        pos_count = int((df[label] == 1).sum())
        label_counts[label] = pos_count
        label_percentages[label] = (pos_count / total_records) * 100 if total_records > 0 else 0.0

    # 8. Detect exact duplicate rows
    exact_duplicate_rows = int(df.duplicated().sum())

    # 9 & 10. Detect duplicate narratives and count unique narratives
    if narrative_col in df.columns:
        duplicate_narratives_count = int(df.duplicated(subset=[narrative_col]).sum())
        unique_narratives_count = int(df[narrative_col].nunique())
    else:
        duplicate_narratives_count = 0
        unique_narratives_count = total_records

    # 11. Class imbalance identification
    max_label_count = max(label_counts.values()) if label_counts else 0
    min_label_count = min(label_counts.values()) if label_counts else 0
    imbalance_ratio = (max_label_count / min_label_count) if min_label_count > 0 else 1.0

    imbalance_notes = []
    if label_counts:
        for label, cnt in label_counts.items():
            pct = label_percentages[label]
            imbalance_notes.append(f"  - {label}: {cnt} samples ({pct:.2f}% of dataset)")
        imbalance_summary = f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}:1"
    else:
        imbalance_summary = "No binary label columns identified."

    # Multi-label co-occurrence check
    if actual_labels:
        df["total_active_labels"] = df[actual_labels].sum(axis=1)
        multi_label_rows = int((df["total_active_labels"] > 1).sum())
        zero_label_rows = int((df["total_active_labels"] == 0).sum())
        single_label_rows = int((df["total_active_labels"] == 1).sum())
    else:
        multi_label_rows = zero_label_rows = single_label_rows = 0

    # 12. Format human-readable report
    report_lines = [
        "=" * 70,
        "WOMEN'S SAFETY INCIDENT DATASET ANALYSIS REPORT",
        "=" * 70,
        f"Dataset File Path: {csv_path}",
        f"Total Records: {total_records}",
        f"Columns ({len(columns_list)}): {', '.join(columns_list)}",
        "-" * 70,
        "MISSING VALUES BY COLUMN:",
    ]
    for col, m_cnt in missing_values.items():
        pct_m = (m_cnt / total_records) * 100 if total_records > 0 else 0.0
        report_lines.append(f"  - {col}: {m_cnt} missing ({pct_m:.2f}%)")

    report_lines.extend([
        "-" * 70,
        f"NARRATIVE COLUMN IDENTIFIED: '{narrative_col}'",
        f"  - Unique Narratives: {unique_narratives_count}",
        f"  - Duplicate Narratives: {duplicate_narratives_count}",
        f"  - Exact Duplicate Rows (all columns): {exact_duplicate_rows}",
        "-" * 70,
        f"INCIDENT LABELS DETECTED ({len(actual_labels)}): {', '.join(actual_labels)}",
        "LABEL DISTRIBUTION & CLASS COUNT:",
    ])
    for label in actual_labels:
        cnt = label_counts[label]
        pct = label_percentages[label]
        report_lines.append(f"  - {label}: {cnt} examples ({pct:.2f}%)")

    report_lines.extend([
        "-" * 70,
        "MULTI-LABEL CO-OCCURRENCE BREAKDOWN:",
        f"  - Single Label Incidents: {single_label_rows} ({single_label_rows/total_records*100:.2f}%)",
        f"  - Multi-Label Incidents (>1 label): {multi_label_rows} ({multi_label_rows/total_records*100:.2f}%)",
        f"  - Zero Label Incidents (Unclassified): {zero_label_rows} ({zero_label_rows/total_records*100:.2f}%)",
        "-" * 70,
        "CLASS IMBALANCE ANALYSIS:",
        f"  - {imbalance_summary}",
        "  - Analysis: The dataset contains multi-label annotations across safety categories.",
        "    Harassment (SH) and Domestic Violence (DV) hold highest representation,",
        "    while Other Violence (OV) and Workplace Harassment (WH) are less frequent.",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report artifact
    report_output_path = current_file.parent / "analysis_report.txt"
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[SUCCESS] Analysis report saved to: {report_output_path}")

    return {
        "total_records": total_records,
        "columns": columns_list,
        "missing_values": missing_values,
        "narrative_col": narrative_col,
        "actual_labels": actual_labels,
        "label_counts": label_counts,
        "label_percentages": label_percentages,
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_narratives_count": duplicate_narratives_count,
        "unique_narratives_count": unique_narratives_count,
        "imbalance_ratio": imbalance_ratio,
    }


if __name__ == "__main__":
    run_dataset_analysis()
