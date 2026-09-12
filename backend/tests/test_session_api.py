"""
Unit and Integration Tests for Backend Session Management API.

Verifies:
- POST /api/session/start creates session and incident submission
- Session fields: session_id, submission_id, messages, answers, predicted_labels, entities, current_question, conversation_status
- Default state initialization (empty lists/dict, status ACTIVE)
- GET /api/session/{session_id} retrieves persisted session state
- Unknown session_id returns 404
- Allowed conversation statuses: ACTIVE, QUESTIONING, PROCESSING, COMPLETED
- Session persistence and state updates via SessionService
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.session import Session as UserSession, ALLOWED_CONVERSATION_STATUSES
from app.models.incident_submission import IncidentSubmission
from app.services.session_service import SessionService


class TestSessionAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.created_session_ids = []

    @classmethod
    def tearDownClass(cls):
        # Clean up database test records created during unit tests
        try:
            for sess_id in cls.created_session_ids:
                cls.db.query(IncidentSubmission).filter(IncidentSubmission.session_id == sess_id).delete()
                cls.db.query(UserSession).filter(UserSession.session_id == sess_id).delete()
            cls.db.commit()
        except Exception:
            cls.db.rollback()
        finally:
            cls.db.close()

    # 1. POST /api/session/start returns 200/201 status code
    def test_01_post_session_start_returns_success(self):
        response = self.client.post("/api/session/start", json={})
        self.assertIn(response.status_code, [200, 201])
        data = response.json()
        self.created_session_ids.append(data["session_id"])

    # 2. Response contains session_id
    def test_02_response_contains_session_id(self):
        response = self.client.post("/api/session/start", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertTrue(data["session_id"].startswith("SESS-"))
        self.created_session_ids.append(data["session_id"])

    # 3. Response contains submission_id
    def test_03_response_contains_submission_id(self):
        response = self.client.post("/api/session/start", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("submission_id", data)
        self.assertTrue(data["submission_id"].startswith("SUB-"))
        self.created_session_ids.append(data["session_id"])

    # 4. Initial messages are empty list
    def test_04_initial_messages_are_empty(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertEqual(data["messages"], [])
        self.created_session_ids.append(data["session_id"])

    # 5. Initial answers are empty list
    def test_05_initial_answers_are_empty(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertEqual(data["answers"], [])
        self.created_session_ids.append(data["session_id"])

    # 6. Initial predicted_labels are empty list
    def test_06_initial_predicted_labels_are_empty(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertEqual(data["predicted_labels"], [])
        self.created_session_ids.append(data["session_id"])

    # 7. Initial entities are empty dict
    def test_07_initial_entities_are_empty(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertEqual(data["entities"], {})
        self.created_session_ids.append(data["session_id"])

    # 8. current_question is null
    def test_08_initial_current_question_is_null(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertIsNone(data["current_question"])
        self.created_session_ids.append(data["session_id"])

    # 9. conversation_status is ACTIVE
    def test_09_initial_conversation_status_is_active(self):
        response = self.client.post("/api/session/start", json={})
        data = response.json()
        self.assertEqual(data["conversation_status"], "ACTIVE")
        self.created_session_ids.append(data["session_id"])

    # 10. GET /api/session/{session_id} returns the same persisted session
    def test_10_get_session_returns_persisted_state(self):
        start_res = self.client.post("/api/session/start", json={})
        session_data = start_res.json()
        sess_id = session_data["session_id"]
        self.created_session_ids.append(sess_id)

        get_res = self.client.get(f"/api/session/{sess_id}")
        self.assertEqual(get_res.status_code, 200)
        get_data = get_res.json()

        self.assertEqual(get_data["session_id"], sess_id)
        self.assertEqual(get_data["submission_id"], session_data["submission_id"])
        self.assertEqual(get_data["conversation_status"], "ACTIVE")

    # 11. Unknown session_id returns 404
    def test_11_unknown_session_id_returns_404(self):
        response = self.client.get("/api/session/SESS-UNKNOWN-999999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    # 12. Session state survives retrieval and updates
    def test_12_session_state_survives_retrieval_and_update(self):
        start_res = self.client.post("/api/session/start", json={})
        session_data = start_res.json()
        sess_id = session_data["session_id"]
        self.created_session_ids.append(sess_id)

        # Update session state via service
        updated = SessionService.update_session(
            self.db,
            sess_id,
            {
                "messages": [{"sender": "user", "text": "Help me"}],
                "predicted_labels": ["Workplace Harassment"],
                "conversation_status": "QUESTIONING",
            },
        )
        self.assertIsNotNone(updated)

        # Fetch via GET endpoint to verify persistence
        get_res = self.client.get(f"/api/session/{sess_id}")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()

        self.assertEqual(data["conversation_status"], "QUESTIONING")
        self.assertEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["text"], "Help me")
        self.assertEqual(data["predicted_labels"], ["Workplace Harassment"])

    # 13. Status values are restricted to ALLOWED_CONVERSATION_STATUSES
    def test_13_status_values_restricted_to_allowed_set(self):
        expected_allowed = {"ACTIVE", "QUESTIONING", "PROCESSING", "COMPLETED"}
        self.assertEqual(ALLOWED_CONVERSATION_STATUSES, expected_allowed)

        start_res = self.client.post("/api/session/start", json={})
        sess_id = start_res.json()["session_id"]
        self.created_session_ids.append(sess_id)

        # Valid updates should work
        for valid_status in ["QUESTIONING", "PROCESSING", "COMPLETED", "ACTIVE"]:
            updated = SessionService.update_session(self.db, sess_id, {"conversation_status": valid_status})
            self.assertEqual(updated.conversation_status, valid_status)

        # Invalid status should raise ValueError
        with self.assertRaises(ValueError):
            SessionService.update_session(self.db, sess_id, {"conversation_status": "INVALID_STATUS_XYZ"})


if __name__ == "__main__":
    unittest.main()
