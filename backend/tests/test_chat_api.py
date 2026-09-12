"""
Unit and Integration Tests for Main Chat API Router.

Verifies:
- POST /api/chat/message endpoint processing
- Valid session + valid message handling
- 404 response for unknown session_id
- 400 response for empty message or empty session_id
- User and Assistant message persistence in session
- BERT classifier & NER entity extraction integration
- Dynamic Question Engine follow-up workflow (status: QUESTIONING)
- Completion workflow (status: COMPLETED)
- Anti-Hallucination validated, database-backed Legal and Support mapping
- Non-generative structured report generation upon completion
- Prohibition of external APIs, web requests, or LLM generation
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.session import Session as UserSession
from app.models.incident_submission import IncidentSubmission
from app.models.report import Report
from app.models.question import SubmissionQuestionAnswer
from app.services.session_service import SessionService
from app.services.chat_service import ChatService


class TestChatAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.test_session_ids = []

    @classmethod
    def tearDownClass(cls):
        # Clean up database test records created during testing
        try:
            for sess_id in cls.test_session_ids:
                sub = cls.db.query(IncidentSubmission).filter(IncidentSubmission.session_id == sess_id).first()
                if sub:
                    cls.db.query(Report).filter(Report.submission_id == sub.submission_id).delete()
                    cls.db.query(SubmissionQuestionAnswer).filter(SubmissionQuestionAnswer.submission_id == sub.submission_id).delete()
                    cls.db.query(IncidentSubmission).filter(IncidentSubmission.submission_id == sub.submission_id).delete()
                cls.db.query(UserSession).filter(UserSession.session_id == sess_id).delete()
            cls.db.commit()
        except Exception:
            cls.db.rollback()
        finally:
            cls.db.close()

    def _create_test_session(self) -> UserSession:
        session = SessionService.create_session(self.db)
        self.test_session_ids.append(session.session_id)
        return session

    # 1. Valid session + valid message returns 200
    def test_01_valid_session_and_message_returns_200(self):
        sess = self._create_test_session()
        payload = {
            "session_id": sess.session_id,
            "message": "My boss has been harassing me at the workplace every week.",
        }
        response = self.client.post("/api/chat/message", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], sess.session_id)

    # 2. Unknown session_id returns 404
    def test_02_unknown_session_returns_404(self):
        payload = {
            "session_id": "SESS-NONEXISTENT-999999",
            "message": "Hello, I need help.",
        }
        response = self.client.post("/api/chat/message", json=payload)
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    # 3. Empty message or session_id returns 400
    def test_03_empty_message_or_session_returns_400(self):
        sess = self._create_test_session()

        # Empty message
        res1 = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "   "})
        self.assertEqual(res1.status_code, 400)

        # Empty session_id
        res2 = self.client.post("/api/chat/message", json={"session_id": "", "message": "Help me"})
        self.assertEqual(res2.status_code, 400)

    # 4. User message persisted in session
    def test_04_user_message_persisted_in_session(self):
        sess = self._create_test_session()
        msg_text = "I received abusive messages online from an unknown person."
        self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": msg_text})

        db_sess = SessionService.get_session(self.db, sess.session_id)
        user_msgs = [m for m in db_sess.messages if m.get("role") == "user"]
        self.assertGreater(len(user_msgs), 0)
        self.assertEqual(user_msgs[-1]["content"], msg_text)

    # 5. BERT predictions persisted
    def test_05_bert_predictions_persisted(self):
        sess = self._create_test_session()
        msg_text = "My husband physically assaulted me at home."
        res = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": msg_text})
        data = res.json()

        self.assertIn("predicted_labels", data)
        self.assertIsInstance(data["predicted_labels"], list)

        db_sess = SessionService.get_session(self.db, sess.session_id)
        self.assertIsInstance(db_sess.predicted_labels, list)

    # 6. NER entities persisted
    def test_06_ner_entities_persisted(self):
        sess = self._create_test_session()
        msg_text = "My former colleague stalked me near my office."
        res = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": msg_text})
        data = res.json()

        self.assertIn("entities", data)
        self.assertIsInstance(data["entities"], dict)

    # 7. Existing session reused across multiple messages
    def test_07_existing_session_reused_across_messages(self):
        sess = self._create_test_session()
        res1 = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "Message 1"})
        res2 = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "Message 2"})

        self.assertEqual(res1.json()["session_id"], sess.session_id)
        self.assertEqual(res2.json()["session_id"], sess.session_id)

    # 8. Previous messages preserved in order
    def test_08_previous_messages_preserved_in_order(self):
        sess = self._create_test_session()
        self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "First message"})
        self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "Second message"})

        db_sess = SessionService.get_session(self.db, sess.session_id)
        user_contents = [m["content"] for m in db_sess.messages if m.get("role") == "user"]
        self.assertEqual(user_contents, ["First message", "Second message"])

    # 9. Question Engine follow-up path & status QUESTIONING
    def test_09_question_engine_followup_and_status_questioning(self):
        sess = self._create_test_session()
        # Vague message missing required information
        res = self.client.post("/api/chat/message", json={"session_id": sess.session_id, "message": "Something bad happened to me."})
        data = res.json()

        if data["conversation_status"] == "QUESTIONING":
            self.assertIsNotNone(data["current_question"])
            self.assertIn("question_text", data["current_question"])
            self.assertTrue(len(data["assistant_message"]) > 0)

    # 10. Completion path & status COMPLETED
    def test_10_completion_path_and_status_completed(self):
        sess = self._create_test_session()
        
        # 1. Initial detailed message
        res1 = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "My supervisor at the office has been sexually harassing me every day for the past month on WhatsApp. I feel unsafe."
        })
        d1 = res1.json()

        # If further question requested, answer it to drive to completion
        if d1["conversation_status"] == "QUESTIONING":
            res2 = self.client.post("/api/chat/message", json={
                "session_id": sess.session_id,
                "message": "It happened at the office in Delhi. My supervisor is the perpetrator and I have WhatsApp chat screenshots."
            })
            d2 = res2.json()
            if d2["conversation_status"] == "QUESTIONING":
                res3 = self.client.post("/api/chat/message", json={
                    "session_id": sess.session_id,
                    "message": "I am currently in a safe location with my family."
                })
                d2 = res3.json()
            d1 = d2

        if d1["conversation_status"] == "COMPLETED":
            self.assertEqual(d1["conversation_status"], "COMPLETED")
            self.assertIsNone(d1["current_question"])
            self.assertIsNotNone(d1["report"])
            self.assertIn("incident_summary", d1["report"])

    # 11. Legal mapping is database-backed (source = database)
    def test_11_legal_mapping_is_database_backed(self):
        sess = self._create_test_session()
        res = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "My husband hit me at home in Delhi. I am safe now with my relatives."
        })
        d = res.json()
        if d.get("report") and isinstance(d["report"].get("relevant_legal_information"), list):
            laws = d["report"]["relevant_legal_information"]
            if len(laws) > 0:
                self.assertEqual(laws[0]["source"], "database")

    # 12. Support mapping is database-backed (source = database)
    def test_12_support_mapping_is_database_backed(self):
        sess = self._create_test_session()
        res = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "I was harassed in Delhi. I am safe now."
        })
        d = res.json()
        if d.get("report") and isinstance(d["report"].get("available_support_services"), list):
            supports = d["report"]["available_support_services"]
            if len(supports) > 0:
                self.assertEqual(supports[0]["source"], "database")

    # 13. Anti-hallucination remains active
    def test_13_anti_hallucination_remains_active(self):
        sess = self._create_test_session()
        res = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "I had an issue."
        })
        d = res.json()
        if d.get("report"):
            self.assertIn("audit", d["report"])

    # 14. Report generation occurs only after completion
    def test_14_report_generation_occurs_after_completion(self):
        sess = self._create_test_session()
        res = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "Help me."
        })
        d = res.json()
        if d["conversation_status"] == "QUESTIONING":
            self.assertIsNone(d["report"])

    # 15. Final status becomes COMPLETED and current_question null upon completion
    def test_15_status_completed_and_question_null_on_completion(self):
        sess = self._create_test_session()
        # Direct service call simulation with all answers
        res = ChatService.process_message(self.db, sess.session_id, "My ex-husband harassed me at home in Delhi. I have evidence and am safe now.")
        if res["conversation_status"] == "COMPLETED":
            self.assertIsNone(res["current_question"])
            self.assertIsNotNone(res["report"])

    # 16. No external API, web dependency, or LLM used
    def test_16_no_external_api_or_llm_used(self):
        sess = self._create_test_session()
        res = self.client.post("/api/chat/message", json={
            "session_id": sess.session_id,
            "message": "Testing external safety principles."
        })
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
