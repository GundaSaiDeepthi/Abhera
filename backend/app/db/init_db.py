import sys
import logging
from app.database import engine, Base
# Import all models to ensure metadata registration
import app.models  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    """
    Safely creates all 10 PostgreSQL database tables using SQLAlchemy metadata.
    Does NOT drop existing tables or insert fake/sample data.
    """
    try:
        logger.info("Connecting to PostgreSQL database and initializing schema...")
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully created/verified all 10 PostgreSQL database tables in womens_safety_db:")
        for table in Base.metadata.tables.keys():
            logger.info(f" - Table: {table}")
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_db()
