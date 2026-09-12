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
logger = logging.getLogger("ClassificationPrep")


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


class ClassificationDataPreparer:
    """Prepares incident dataset into Train/Val/Test multi-label splits for BERT fine-tuning."""

    LABEL_COLUMNS = ["DV", "SH", "ST", "CA", "WH", "OV"]

    def __init__(self, csv_path: Path, output_dir: Path):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        logger.info(f"Loading incident dataset from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        initial_count = len(df)

        narrative_col = "narrative_text" if "narrative_text" in df.columns else "narrative_text"

        valid_mask = df[narrative_col].notnull() & (df[narrative_col].astype(str).str.strip() != "")
        excluded_records = df[~valid_mask]
        excluded_count = len(excluded_records)

        if excluded_count > 0:
            logger.warning(
                f"Excluded {excluded_count} record(s) due to empty narrative text."
            )
        else:
            logger.info("Zero records excluded. All narrative texts are valid.")

        clean_df = df[valid_mask].copy()

        # Multi-hot vector representation
        clean_df["label_vector"] = clean_df[self.LABEL_COLUMNS].values.tolist()

        active_labels_list = []
        for _, row in clean_df.iterrows():
            active = [col for col in self.LABEL_COLUMNS if row[col] == 1]
            active_labels_list.append(active)
        clean_df["active_labels"] = active_labels_list

        # Export full processed CSV artifact
        processed_path = self.output_dir / "processed_classification_data.csv"
        clean_df.to_csv(processed_path, index=False)
        logger.info(f"Saved processed multi-label dataset to: {processed_path}")

        # Compute pos_weights for BCEWithLogitsLoss class imbalance handling
        total_samples = len(clean_df)
        pos_counts = clean_df[self.LABEL_COLUMNS].sum().to_dict()
        pos_weights = {}
        for col in self.LABEL_COLUMNS:
            pos = pos_counts[col]
            neg = total_samples - pos
            pos_weights[col] = float(neg / pos) if pos > 0 else 1.0

        # Label Mapping config artifact
        label_mapping = {
            "labels": self.LABEL_COLUMNS,
            "num_labels": len(self.LABEL_COLUMNS),
            "label2id": {label: idx for idx, label in enumerate(self.LABEL_COLUMNS)},
            "id2label": {idx: label for idx, label in enumerate(self.LABEL_COLUMNS)},
            "pos_weights": pos_weights,
            "pos_counts": pos_counts,
            "total_samples": total_samples,
            "problem_type": "multi_label_classification",
        }
        mapping_path = self.output_dir / "label_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)
        logger.info(f"Saved label mapping & imbalance config to: {mapping_path}")

        # Reproducible Train (70%) / Val (15%) / Test (15%) split using fixed seed
        np.random.seed(seed)
        shuffled_df = clean_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n_samples = len(shuffled_df)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_df = shuffled_df.iloc[:train_end].copy()
        val_df = shuffled_df.iloc[train_end:val_end].copy()
        test_df = shuffled_df.iloc[val_end:].copy()

        train_path = self.output_dir / "train_data.csv"
        val_path = self.output_dir / "val_data.csv"
        test_path = self.output_dir / "test_data.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Train split saved to: {train_path} ({len(train_df)} samples)")
        logger.info(f"Validation split saved to: {val_path} ({len(val_df)} samples)")
        logger.info(f"Test split saved to: {test_path} ({len(test_df)} samples)")

        # Summary verification report
        logger.info("==================================================")
        logger.info("CLASSIFICATION PREPARATION SUMMARY")
        logger.info("==================================================")
        logger.info(f"  Initial Dataset Count: {initial_count}")
        logger.info(f"  Excluded Records: {excluded_count}")
        logger.info(f"  Final Valid Records: {len(clean_df)}")
        logger.info(f"  Problem Formulation: Multi-Label Classification ({len(self.LABEL_COLUMNS)} categories)")
        logger.info(f"  Train Split Size: {len(train_df)} ({len(train_df)/len(clean_df)*100:.1f}%)")
        logger.info(f"  Val Split Size: {len(val_df)} ({len(val_df)/len(clean_df)*100:.1f}%)")
        logger.info(f"  Test Split Size: {len(test_df)} ({len(test_df)/len(clean_df)*100:.1f}%)")
        logger.info("==================================================")

        return {
            "initial_count": initial_count,
            "excluded_count": excluded_count,
            "processed_count": len(clean_df),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "labels": self.LABEL_COLUMNS,
        }


def main():
    try:
        csv_path = resolve_dataset_path()
        output_dir = csv_path.parent.parent / "ml" / "preprocessing"
        preparer = ClassificationDataPreparer(csv_path, output_dir)
        preparer.prepare()
    except Exception as e:
        logger.error(f"Classification preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
