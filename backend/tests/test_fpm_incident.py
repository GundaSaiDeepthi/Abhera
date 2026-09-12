"""
Unit and Integration Tests for False Promise of Marriage Incident Category.

Verifies:
- BERT classification of "A falsely promised to marry B."
- NER entity extraction of "A" as PERP_REL
- Dynamic Question Engine requirement determination
- Legal Mapping fallback message ("Information not available in the provided knowledge base.")
- Support Mapping database service retrieval
- End-to-end chat API pipeline processing
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
from ml.inference.predict_bert import BERTIncidentPredictor
from ml.inference.predict_ner import NEREntityPredictor
from app.services.legal_mapping import map_legal_provisions, NO_MATCH_MESSAGE as LEGAL_NO_MATCH_MESSAGE
from app.services.support_mapping import map_support_services
from app.services.report_generator import generate_incident_report


class TestFalsePromiseOfMarriageIncident(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.test_session_ids = []

    @classmethod
    def tearDownClass(cls):
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

    def test_bert_prediction_for_fpm(self):
        predictor = BERTIncidentPredictor()
        text = "A falsely promised to marry B."
        result = predictor.predict(text)
        labels = [l["name"] for l in result["predicted_labels"]]
        self.assertIn("False Promise of Marriage", labels)

    def test_ner_extraction_for_fpm(self):
        predictor = NEREntityPredictor()
        text = "A falsely promised to marry B."
        entities = predictor.predict(text)
        perp_entities = [e["text"] for e in entities if e["label"] == "PERP_REL"]
        self.assertTrue(len(perp_entities) > 0)
        self.assertIn("A", perp_entities)

    def test_legal_mapping_fallback_for_fpm(self):
        result = map_legal_provisions(predicted_labels=["False Promise of Marriage"], db=self.db)
        self.assertFalse(result["matched"])
        self.assertEqual(result["legal_information"], [])
        self.assertEqual(result["message"], LEGAL_NO_MATCH_MESSAGE)

    def test_support_mapping_for_fpm(self):
        result = map_support_services(predicted_labels=["False Promise of Marriage"], db=self.db)
        self.assertTrue(result["matched"])
        self.assertTrue(len(result["support_services"]) > 0)

    def test_end_to_end_chat_api_fpm(self):
        sess = self._create_test_session()
        payload = {
            "session_id": sess.session_id,
            "message": "A falsely promised to marry B."
        }
        res = self.client.post("/api/chat/message", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("predicted_labels"), ["False Promise of Marriage"])
        self.assertIn(data.get("conversation_status"), ["QUESTIONING", "COMPLETED"])

    def test_disambiguation_case_a(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("A falsely promised to marry B.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertEqual(labels, ["False Promise of Marriage"])

    def test_disambiguation_case_b(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("He falsely promised to marry me.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertEqual(labels, ["False Promise of Marriage"])

    def test_disambiguation_case_c(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("He promised to marry me but later refused.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertEqual(labels, ["False Promise of Marriage"])

    def test_disambiguation_case_d(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("My husband beats me.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)
        self.assertNotIn("False Promise of Marriage", labels)

    def test_disambiguation_case_e(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("My husband threatens and physically abuses me.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("Domestic Violence", labels)
        self.assertNotIn("False Promise of Marriage", labels)

    def test_disambiguation_case_f_mixed(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("He falsely promised to marry me and physically abused me.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("False Promise of Marriage", labels)
        self.assertIn("Domestic Violence", labels)

    def test_disambiguation_case_g_mixed(self):
        predictor = BERTIncidentPredictor()
        res = predictor.predict("He promised to marry me and later beat me.")
        labels = [l["name"] for l in res["predicted_labels"]]
        self.assertIn("False Promise of Marriage", labels)
        self.assertIn("Domestic Violence", labels)


if __name__ == "__main__":
    unittest.main()
