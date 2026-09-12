import logging
import sys
from pathlib import Path
from sqlalchemy import text

# Ensure backend root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.database import engine, Base
import app.models  # noqa: F401
from app.models.question import Question, SubmissionQuestionAnswer
from app.db.seed_questions import seed_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MigrateQuestions")


def migrate_questions_table():
    """
    Applies PostgreSQL migration for the dynamic questions table and submission_question_answers table.
    Safely replaces legacy questions schema with the new required schema.
    """
    try:
        with engine.connect() as conn:
            logger.info("Safely recreating questions and submission_question_answers PostgreSQL tables...")
            conn.execute(text("DROP TABLE IF EXISTS submission_question_answers CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS questions CASCADE;"))
            conn.commit()
            logger.info("Dropped legacy questions tables.")

        # Recreate tables defined in SQLAlchemy metadata
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully created questions and submission_question_answers PostgreSQL tables.")

        # Seed initial question bank
        seed_questions()
        logger.info("Seeding complete.")

    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    migrate_questions_table()
