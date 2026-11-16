"""
ABHERA — Production Database Initializer & Seeder for Supabase PostgreSQL

SAFETY ARCHITECTURE & OPERATIONAL BOUNDARIES:
---------------------------------------------
1. URL SAFETY ENFORCEMENT:
   - Strictly validates target database URL.
   - Automatically REJECTS SQLite, localhost, 127.0.0.1, 0.0.0.0, and loopback dev hosts.
   - Requires a valid remote production PostgreSQL endpoint (e.g. Supabase pooler).
   - Testing bypass only available via explicit --allow-local-test flag.

2. EXPLICIT CONFIRMATION MANDATE:
   - Requires explicit --confirm flag to execute against production.
   - In interactive mode without --confirm, prompts user for strict typed confirmation.
   - In non-interactive mode without --confirm, aborts immediately.

3. NON-DESTRUCTIVE TABLE CREATION:
   - Uses SQLAlchemy `Base.metadata.create_all(bind=engine)`.
   - Executes `CREATE TABLE IF NOT EXISTS`.
   - NEVER issues DROP TABLE, ALTER TABLE, or schema destruction.

4. SURVIVOR & OPERATIONAL DATA PRESERVATION:
   - The following 8 operational survivor tables are NEVER cleared, deleted, or dropped:
     * users
     * sessions
     * incident_submissions
     * conversation_messages
     * predictions
     * entities
     * submission_question_answers
     * reports
   - Existing user accounts, active sessions, incident reports, and chat histories remain 100% untouched.

5. REFERENCE TABLE REFRESH & QUESTION IDEMPOTENCY:
   - Reference Tables: `laws` (32 statutory sections) and `support_services` (12 verified helplines)
     are refreshed from ground-truth CSVs (`data/laws.csv` and `data/support_services.csv`) via
     DatasetImporter. These static lookup tables replace reference contents on fresh seed.
   - Question Seeding: `seed_questions()` matches by unique `question_id`. Existing questions are
     updated in-place; new questions are inserted. Repeated execution NEVER creates duplicates.

Usage:
  python scripts/init_production_db.py --db-url "postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require" --confirm
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("InitProductionDB")


def validate_production_db_url(url: str, allow_local_test: bool = False) -> None:
    """
    Validates that the provided URL is a genuine remote production PostgreSQL endpoint.
    Strictly rejects SQLite, localhost, 127.0.0.1, and local development hostnames.
    """
    if not url or not isinstance(url, str):
        raise ValueError("DATABASE_URL must be a non-empty string.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme.startswith("sqlite"):
        if not allow_local_test:
            raise ValueError(
                "CRITICAL DATABASE SAFETY VIOLATION: SQLite database URLs are strictly rejected "
                "for production initialization. Provide a remote PostgreSQL endpoint (e.g., Supabase)."
            )
        return

    if not scheme.startswith("postgresql"):
        raise ValueError(
            f"CRITICAL DATABASE SAFETY VIOLATION: Database scheme '{parsed.scheme}' is unsupported. "
            "Production initialization requires a valid 'postgresql://' URL."
        )

    host = (parsed.hostname or "").lower()
    dev_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}

    if host in dev_hosts or host.endswith(".local") or host.endswith(".lan"):
        if not allow_local_test:
            raise ValueError(
                f"CRITICAL DATABASE SAFETY VIOLATION: Localhost/development host '{host}' is strictly rejected! "
                "Production initialization must NEVER target local development databases. "
                "Provide a valid remote production endpoint (e.g., Supabase pooler: aws-0-*.pooler.supabase.com)."
            )


def mask_url_credentials(url: str) -> str:
    """Masks database password in connection string for secure logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:********@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return url.replace(parsed.netloc, netloc)
    except Exception:
        pass
    return url.split("@")[-1] if "@" in url else "specified URL"


def run_production_init(database_url: str, allow_local_test: bool = False):
    logger.info("=================================================================")
    logger.info("ABHERA — PRODUCTION DATABASE INITIALIZATION & SAFETY AUDIT")
    logger.info("=================================================================")

    # 1. Validate Target Database URL Safety
    validate_production_db_url(database_url, allow_local_test=allow_local_test)
    logger.info(f"Target Database Endpoint (Masked): {mask_url_credentials(database_url)}")

    # 2. Configure Database Environment Variable Before Importing ORM
    os.environ["DATABASE_URL"] = database_url

    import app.config
    app.config.settings.DATABASE_URL = database_url

    from sqlalchemy import create_engine, inspect
    from app.database import Base, engine as app_engine, SessionLocal
    import app.models  # Register all 11 ORM models with metadata
    from app.db.seed_questions import seed_questions
    from app.services.data_importer import DatasetImporter
    from app.models.question import Question
    from app.models.law import Law
    from app.models.support_service import SupportService

    try:
        # 3. Non-Destructive Schema Creation (11 application tables)
        logger.info("Ensuring all 11 application tables exist in schema (CREATE TABLE IF NOT EXISTS)...")
        Base.metadata.create_all(bind=app_engine)
        inspector = inspect(app_engine)
        existing_tables = sorted(inspector.get_table_names())
        logger.info(f"Verified tables in target schema ({len(existing_tables)} tables): {existing_tables}")

        # 4. Idempotent Question Seeding (15 dynamic questions)
        logger.info("Seeding dynamic follow-up questions (Idempotent Upsert by question_id)...")
        db = SessionLocal()
        try:
            seed_questions(db=db)
        finally:
            db.close()

        # 5. Reference Data Import: Laws & Support Services (Static Lookup Tables Only)
        data_dir = PROJECT_ROOT / "data"
        if data_dir.exists() and (data_dir / "laws.csv").exists():
            logger.info(f"Importing statutory laws and emergency helplines from: {data_dir}")
            logger.info("NOTE: Reference tables 'laws' and 'support_services' are refreshed from authoritative CSV datasets.")
            logger.info("NOTE: Operational tables (users, sessions, incidents, messages, reports) are 100% UNTOUCHED.")
            importer = DatasetImporter(data_dir=data_dir)
            laws_count = importer.process_laws_csv()
            support_count = importer.process_support_services_csv()
            logger.info(f"Successfully loaded {laws_count} laws and {support_count} support services.")
        else:
            logger.warning(f"Data directory not found at {data_dir}. Skipping CSV reference dataset import.")

        # 6. Verify Database State & Row Counts
        db_verify = SessionLocal()
        try:
            q_count = db_verify.query(Question).count()
            l_count = db_verify.query(Law).count()
            s_count = db_verify.query(SupportService).count()

            logger.info("=================================================================")
            logger.info("PRODUCTION DATABASE INITIALIZATION COMPLETE:")
            logger.info(f" - Total Database Tables     : {len(existing_tables)} (Expected: 11)")
            logger.info(f" - Dynamic Questions Seeded  : {q_count} (Expected: 15, Unique)")
            logger.info(f" - Statutory Laws Seeded     : {l_count} (Expected: 32)")
            logger.info(f" - Support Helplines Seeded  : {s_count} (Expected: 12)")
            logger.info("Status: PRODUCTION DATABASE INITIALIZED SUCCESSFULLY & SAFELY")
            logger.info("=================================================================")
        finally:
            db_verify.close()

    except Exception as e:
        logger.error(f"Error during production database initialization: {e}")
        raise e
    finally:
        app_engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABHERA Production Database Initializer")
    parser.add_argument("--db-url", type=str, default=None, help="Target PostgreSQL connection URL")
    parser.add_argument("--confirm", action="store_true", help="Explicit confirmation to execute against production")
    parser.add_argument("--allow-local-test", action="store_true", help="Testing flag to allow local SQLite dry-runs")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL must be provided via --db-url argument or DATABASE_URL environment variable.")
        sys.exit(1)

    # Confirmation Gate
    if not args.confirm and not args.allow_local_test:
        if sys.stdin.isatty():
            print("\n" + "=" * 65)
            print("WARNING: PRODUCTION DATABASE INITIALIZATION CONFIRMATION")
            print("=" * 65)
            print(f"Target: {mask_url_credentials(db_url)}")
            print("This operation will create 11 application tables and seed statutory reference data.")
            user_input = input("Type 'INITIALIZE-ABHERA-PRODUCTION' to proceed: ").strip()
            if user_input != "INITIALIZE-ABHERA-PRODUCTION":
                logger.error("Initialization aborted: confirmation phrase did not match.")
                sys.exit(1)
        else:
            logger.error(
                "CRITICAL SAFETY ABORT: Production database initialization requires the explicit '--confirm' flag "
                "when executed in non-interactive/script mode."
            )
            sys.exit(1)

    run_production_init(database_url=db_url, allow_local_test=args.allow_local_test)
