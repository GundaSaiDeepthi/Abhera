import logging
import sys
from pathlib import Path
from sqlalchemy.orm import Session as DBSession

from app.database import engine, SessionLocal
from app.models.question import Question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SeedQuestions")

INITIAL_QUESTIONS = [
    {
        "question_id": "Q_SAFETY_01",
        "category": "General",
        "question_text": "Are you currently in immediate physical danger, or do you need emergency support right now?",
        "required_entity": "SAFETY_STATUS",
        "priority": 100,
        "active": True,
    },
    {
        "question_id": "Q_PERP_REL_01",
        "category": "General",
        "question_text": "What is your relationship to the person involved (e.g., spouse/partner, family member, landlord, colleague, acquaintance, or stranger)?",
        "required_entity": "PERP_REL",
        "priority": 90,
        "active": True,
    },
    {
        "question_id": "Q_LOCATION_01",
        "category": "Sexual Harassment",
        "question_text": "Where did this incident take place (e.g., workplace, public transport, educational institution, online, or home)?",
        "required_entity": "LOCATION",
        "priority": 85,
        "active": True,
    },
    {
        "question_id": "Q_TIME_FREQ_01",
        "category": "Stalking",
        "question_text": "How long or how frequently has this behavior been occurring (e.g., daily, over the past week/month)?",
        "required_entity": "TIME_FREQ",
        "priority": 85,
        "active": True,
    },
    {
        "question_id": "Q_PLATFORM_01",
        "category": "Cyber Harassment",
        "question_text": "Which platform, app, or communication channel was used during this incident (e.g., WhatsApp, Instagram, email, phone calls)?",
        "required_entity": "PLATFORM",
        "priority": 85,
        "active": True,
    },
    {
        "question_id": "Q_DETAILS_01",
        "category": "General",
        "question_text": "Could you describe what specifically happened during the incident?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 70,
        "active": True,
    },
    {
        "question_id": "Q_DV_DETAILS_01",
        "category": "Domestic Violence",
        "question_text": "Has there been any physical violence, threats, or financial/emotional abuse by a family member or partner?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_ST_DETAILS_01",
        "category": "Stalking",
        "question_text": "Has the person been following you physically, tracking your location, or sending unwanted repeated messages?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_WH_LOCATION_01",
        "category": "Workplace Harassment",
        "question_text": "Did this occur at your workplace, during work-related events, or through work communication channels?",
        "required_entity": "LOCATION",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_SH_DETAILS_01",
        "category": "Sexual Harassment",
        "question_text": "Did the incident involve uninvited physical contact, sexual remarks, or coercion?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_CA_PLATFORM_01",
        "category": "Cyber Harassment",
        "question_text": "On which online social media or messaging platform did the harassment occur?",
        "required_entity": "PLATFORM",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_CA_DETAILS_01",
        "category": "Cyber Harassment",
        "question_text": "Did the harassment involve unwanted messages, leaked personal media, fake profiles, or online threats?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_WH_DETAILS_01",
        "category": "Workplace Harassment",
        "question_text": "Did the workplace incident involve professional retaliation, unwelcome comments, intimidation, or harassment by a supervisor or colleague?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
    {
        "question_id": "Q_OV_DETAILS_01",
        "category": "Other Harassment",
        "question_text": "Please provide any additional details about the incident or threats you experienced.",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 75,
        "active": True,
    },
    {
        "question_id": "Q_FPM_DETAILS_01",
        "category": "False Promise of Marriage",
        "question_text": "Did the incident involve a false promise of marriage, deceitful engagement, or breach of promise under false pretenses?",
        "required_entity": "INCIDENT_DETAILS",
        "priority": 88,
        "active": True,
    },
]


def seed_questions(db: DBSession = None):
    """
    Seeds initial dynamic questions into the PostgreSQL database.
    Upserts without duplicating existing records.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        inserted = 0
        updated = 0
        for q_data in INITIAL_QUESTIONS:
            existing = db.query(Question).filter(Question.question_id == q_data["question_id"]).first()
            if existing:
                existing.category = q_data["category"]
                existing.question_text = q_data["question_text"]
                existing.required_entity = q_data["required_entity"]
                existing.priority = q_data["priority"]
                existing.active = q_data["active"]
                updated += 1
            else:
                new_q = Question(**q_data)
                db.add(new_q)
                inserted += 1

        db.commit()
        logger.info(f"Seed questions completed: {inserted} inserted, {updated} updated (Total: {len(INITIAL_QUESTIONS)} questions).")
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding questions: {e}")
        raise e
    finally:
        if close_session:
            db.close()


if __name__ == "__main__":
    seed_questions()
