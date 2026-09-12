import logging
import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.database import engine, Base
import app.models  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MigrateSessions")


def migrate_sessions_table():
    """
    Safely adds new session management columns to the existing PostgreSQL 'sessions' table
    if they do not already exist. Does NOT drop existing tables or delete existing data.
    """
    try:
        with engine.connect() as conn:
            logger.info("Verifying/adding session management columns in PostgreSQL 'sessions' table...")

            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS submission_id VARCHAR(64);"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS messages JSON DEFAULT '[]'::json;"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS answers JSON DEFAULT '[]'::json;"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS predicted_labels JSON DEFAULT '[]'::json;"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS entities JSON DEFAULT '{}'::json;"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS current_question JSON;"))
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS conversation_status VARCHAR(30) DEFAULT 'ACTIVE';"))

            conn.commit()
            logger.info("Successfully updated PostgreSQL 'sessions' table schema.")

        Base.metadata.create_all(bind=engine)
        logger.info("Database schema migration completed successfully.")

    except Exception as e:
        logger.error(f"Error migrating sessions table schema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    migrate_sessions_table()
