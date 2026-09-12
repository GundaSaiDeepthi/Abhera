import logging
from pathlib import Path
import sys
import pandas as pd

from app.database import SessionLocal, engine
from app.models.law import Law
from app.models.support_service import SupportService
import app.models  # Ensure metadata registration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DataImporter")


def resolve_data_directory() -> Path:
    """Finds the data/ directory path regardless of current working directory."""
    current_file = Path(__file__).resolve()
    # Possible paths: project_root/data or cwd/data or ../data
    candidates = [
        current_file.parent.parent.parent.parent / "data",
        current_file.parent.parent.parent / "data",
        Path.cwd() / "data",
        Path.cwd().parent / "data",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the data/ directory. Searched in: "
        f"{[str(c) for c in candidates]}"
    )


class DatasetImporter:
    """Dataset importer service that validates, inspects, and imports project CSV datasets."""

    REQUIRED_LAWS_COLUMNS = [
        "act_name",
        "section_number",
        "section_text",
        "applicable_label",
    ]
    REQUIRED_SUPPORT_COLUMNS = [
        "service_type",
        "name",
        "contact_number",
        "state",
        "district",
        "applicable_label",
    ]
    REQUIRED_INCIDENT_COLUMNS = [
        "id",
        "narrative_text",
        "DV",
        "SH",
        "ST",
        "CA",
        "WH",
        "OV",
        "ipc_bns_section",
        "evidence_notes",
        "source",
    ]
    LABEL_COLUMNS = ["DV", "SH", "ST", "CA", "WH", "OV"]

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.stats = {}

    def _validate_schema(
        self, df: pd.DataFrame, expected_columns: list[str], dataset_name: str
    ) -> bool:
        """Validates that all required columns exist in the DataFrame."""
        actual_cols = set(df.columns)
        expected_cols = set(expected_columns)
        missing = expected_cols - actual_cols
        if missing:
            logger.error(
                f"[{dataset_name}] SCHEMA VALIDATION FAILED! Missing required"
                f" columns: {sorted(list(missing))}"
            )
            logger.error(
                f"[{dataset_name}] Actual columns present in file:"
                f" {list(df.columns)}"
            )
            return False
        logger.info(
            f"[{dataset_name}] Schema validation passed. All"
            f" {len(expected_columns)} required columns present."
        )
        return True

    def _inspect_dataframe(
        self, df: pd.DataFrame, dataset_name: str, label_col: str = None
    ) -> dict:
        """Inspects DataFrame statistics: counts, missing values, duplicates, and unique labels."""
        total_records = len(df)
        col_names = list(df.columns)
        missing_counts = df.isnull().sum().to_dict()
        duplicate_count = int(df.duplicated().sum())

        unique_labels = []
        if label_col and label_col in df.columns:
            unique_labels = sorted(
                [str(x) for x in df[label_col].dropna().unique()]
            )

        stat_info = {
            "dataset_name": dataset_name,
            "total_records": total_records,
            "columns": col_names,
            "missing_values": missing_counts,
            "duplicate_count": duplicate_count,
            "unique_labels": unique_labels,
        }
        self.stats[dataset_name] = stat_info

        logger.info(f"--- Dataset Statistics: {dataset_name} ---")
        logger.info(f"  Total Records: {total_records}")
        logger.info(f"  Columns: {col_names}")
        logger.info(f"  Exact Duplicates: {duplicate_count}")
        logger.info(f"  Missing Values: {missing_counts}")
        if unique_labels:
            logger.info(f"  Unique Labels: {unique_labels}")

        return stat_info

    def process_laws_csv(self) -> int:
        """Processes laws.csv and imports records into PostgreSQL laws table."""
        csv_path = self.data_dir / "laws.csv"
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        if not self._validate_schema(df, self.REQUIRED_LAWS_COLUMNS, "laws.csv"):
            return 0

        self._inspect_dataframe(df, "laws.csv", label_col="applicable_label")

        clean_df = df.drop_duplicates().copy()

        db = SessionLocal()
        imported_count = 0
        try:
            existing_count = db.query(Law).count()
            if existing_count > 0:
                logger.info(
                    f"[laws.csv] PostgreSQL table 'laws' already contains {existing_count} records."
                    " Clearing table before fresh import of ground-truth dataset..."
                )
                db.query(Law).delete()
                db.commit()

            laws_objects = []
            for _, row in clean_df.iterrows():
                law_obj = Law(
                    act_name=str(row["act_name"]).strip(),
                    section_number=str(row["section_number"]).strip(),
                    section_text=str(row["section_text"]).strip(),
                    applicable_label=str(row["applicable_label"]).strip(),
                )
                laws_objects.append(law_obj)

            db.bulk_save_objects(laws_objects)
            db.commit()
            imported_count = len(laws_objects)
            logger.info(
                f"[laws.csv] SUCCESSFULLY IMPORTED {imported_count} legal records into PostgreSQL database."
            )
        except Exception as e:
            db.rollback()
            logger.error(f"[laws.csv] Database import error: {e}")
            raise
        finally:
            db.close()

        return imported_count

    def process_support_services_csv(self) -> int:
        """Processes support_services.csv and imports records into PostgreSQL support_services table."""
        csv_path = self.data_dir / "support_services.csv"
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        if not self._validate_schema(
            df, self.REQUIRED_SUPPORT_COLUMNS, "support_services.csv"
        ):
            return 0

        self._inspect_dataframe(
            df, "support_services.csv", label_col="applicable_label"
        )

        clean_df = df.drop_duplicates().copy()

        db = SessionLocal()
        imported_count = 0
        try:
            existing_count = db.query(SupportService).count()
            if existing_count > 0:
                logger.info(
                    "[support_services.csv] PostgreSQL table 'support_services'"
                    f" already contains {existing_count} records. Clearing table"
                    " before fresh import of ground-truth dataset..."
                )
                db.query(SupportService).delete()
                db.commit()

            support_objects = []
            for _, row in clean_df.iterrows():
                svc_obj = SupportService(
                    service_type=str(row["service_type"]).strip(),
                    name=str(row["name"]).strip(),
                    contact_number=str(row["contact_number"]).strip(),
                    state=str(row["state"]).strip(),
                    district=str(row["district"]).strip(),
                    applicable_label=str(row["applicable_label"]).strip(),
                )
                support_objects.append(svc_obj)

            db.bulk_save_objects(support_objects)
            db.commit()
            imported_count = len(support_objects)
            logger.info(
                "[support_services.csv] SUCCESSFULLY IMPORTED"
                f" {imported_count} support service records into PostgreSQL"
                " database."
            )
        except Exception as e:
            db.rollback()
            logger.error(f"[support_services.csv] Database import error: {e}")
            raise
        finally:
            db.close()

        return imported_count

    def process_womens_safety_dataset_csv(self) -> bool:
        """Validates womens_safety_dataset.csv and prepares dataset statistics for ML pipeline."""
        csv_path = self.data_dir / "womens_safety_dataset.csv"
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            return False

        df = pd.read_csv(csv_path)
        if not self._validate_schema(
            df, self.REQUIRED_INCIDENT_COLUMNS, "womens_safety_dataset.csv"
        ):
            return False

        self._inspect_dataframe(df, "womens_safety_dataset.csv")

        # Multi-label verification and statistics
        logger.info(
            "--- Multi-Label Distribution Statistics (womens_safety_dataset.csv) ---"
        )
        label_distribution = {}
        for label_col in self.LABEL_COLUMNS:
            count_positive = int((df[label_col] == 1).sum())
            label_distribution[label_col] = count_positive
            logger.info(f"  Label '{label_col}': {count_positive} positive samples")

        # Verify text length stats
        df["narrative_length"] = df["narrative_text"].astype(str).str.len()
        avg_len = float(df["narrative_length"].mean())
        max_len = int(df["narrative_length"].max())
        min_len = int(df["narrative_length"].min())

        logger.info("--- Narrative Text Preparation for ML Pipeline ---")
        logger.info(
            f"  Avg Character Length: {avg_len:.1f} | Min: {min_len} | Max: {max_len}"
        )
        logger.info(
            "  Dataset validated successfully and prepared for multi-label BERT fine-tuning & NER tokenization."
        )

        return True

    def run_all(self):
        """Runs import process for all three datasets."""
        logger.info("==================================================")
        logger.info("STARTING DATASET IMPORT & VALIDATION PHASE (PART 8)")
        logger.info(f"Data Directory Path: {self.data_dir}")
        logger.info("==================================================")

        laws_imported = self.process_laws_csv()
        support_imported = self.process_support_services_csv()
        ml_prep_success = self.process_womens_safety_dataset_csv()

        logger.info("==================================================")
        logger.info("DATASET IMPORT & VALIDATION SUMMARY")
        logger.info("==================================================")
        logger.info(f"  Laws Records Imported to PostgreSQL: {laws_imported}")
        logger.info(
            f"  Support Service Records Imported to PostgreSQL: {support_imported}"
        )
        logger.info(
            f"  ML Incident Dataset Prepared: {'SUCCESS' if ml_prep_success else 'FAILED'}"
        )
        logger.info("==================================================")


def main():
    try:
        data_dir = resolve_data_directory()
        importer = DatasetImporter(data_dir)
        importer.run_all()
    except Exception as e:
        logger.error(f"Dataset import process failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
