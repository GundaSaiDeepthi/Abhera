"""
Tests for the Input Relevance Validation layer.

Unit tests  : app.services.input_validator.validate_input (every status, required cases A-H)
Integration : POST /api/chat/message — non-incident inputs must not trigger BERT labels,
              NER, question engine, legal/support retrieval or report generation; valid
              incidents must still proceed through the original pipeline.
Preservation: the production BERT model's six labels / thresholds are unchanged.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

from app.services import input_validator as iv
from app.services.input_validator import (
    validate_input,
    INVALID_INPUT, OUT_OF_SCOPE, INSUFFICIENT_INFORMATION, UNCERTAIN_CLASSIFICATION, VALID_INCIDENT,
)

CASE_A = "Today I went to the market to buy vegetables."
CASE_B = "Hello."
CASE_D = "I am facing a problem."
CASE_E = "Someone has been following me every day despite my requests to stop."
CASE_F = "My colleague keeps sending me unwanted messages at work."
CASE_G = "My husband threatened to hurt me yesterday."
CASE_H = "Something happened with my neighbour and I feel uncomfortable."

OOS_MSG = ("The description does not appear to contain information related to harassment or women's safety. "
           "Please describe an incident so that ABHERA can analyse it.")
INSUF_MSG = ("Could you provide a little more information about the situation? "
             "This will help ABHERA understand whether assistance is relevant.")
UNCERT_MSG = ("The information provided is not sufficient to identify the nature of the incident. "
              "Could you clarify what happened?")


class TestInputValidatorUnit(unittest.TestCase):

    def test_A_market_out_of_scope(self):
        r = validate_input(CASE_A)
        self.assertEqual(r["status"], OUT_OF_SCOPE)
        self.assertFalse(r["proceed_with_analysis"])
        self.assertEqual(r["message"], OOS_MSG)

    def test_B_greeting_out_of_scope(self):
        r = validate_input(CASE_B)
        self.assertEqual(r["status"], OUT_OF_SCOPE)
        self.assertTrue(r["is_greeting"])
        self.assertFalse(r["proceed_with_analysis"])

    def test_C_empty_invalid(self):
        for t in ["", "   ", None, "???", "..."]:
            self.assertEqual(validate_input(t)["status"], INVALID_INPUT, t)

    def test_D_insufficient(self):
        r = validate_input(CASE_D)
        self.assertEqual(r["status"], INSUFFICIENT_INFORMATION)
        self.assertEqual(r["message"], INSUF_MSG)

    def test_E_F_G_valid(self):
        for t in (CASE_E, CASE_F, CASE_G):
            r = validate_input(t)
            self.assertEqual(r["status"], VALID_INCIDENT, t)
            self.assertTrue(r["proceed_with_analysis"])

    def test_H_ambiguous_uncertain(self):
        r = validate_input(CASE_H)
        self.assertIn(r["status"], (UNCERTAIN_CLASSIFICATION, INSUFFICIENT_INFORMATION))
        self.assertEqual(r["status"], UNCERTAIN_CLASSIFICATION)
        self.assertEqual(r["message"], UNCERT_MSG)

    def test_common_words_do_not_make_incident(self):
        for t in ["I had a great dinner with my friend at the office party.",
                  "My friend and I went home from the office.",
                  "This person at the market sells good mangoes.",
                  "My mother cooked dinner at home today.",
                  "I hit my head on the door today.",
                  "My brother hit the ball very hard in cricket."]:
            self.assertNotEqual(validate_input(t)["status"], VALID_INCIDENT, t)

    def test_bert_alone_never_upgrades_unrelated_text(self):
        fake = mock.Mock(return_value={"all_scores": {"DV": 0.99}, "thresholds": {"DV": 0.73}})
        self.assertEqual(validate_input(CASE_A, bert_predict=fake)["status"], OUT_OF_SCOPE)
        fake.assert_not_called()

    def test_bert_used_for_harm_without_clear_involvement(self):
        text = "Abusive messages were sent from an unknown account."
        confident = mock.Mock(return_value={"all_scores": {"CA": 0.95}, "thresholds": {"CA": 0.45}})
        weak = mock.Mock(return_value={"all_scores": {"CA": 0.50}, "thresholds": {"CA": 0.45}})
        self.assertEqual(validate_input(text, bert_predict=confident)["status"], VALID_INCIDENT)
        self.assertEqual(validate_input(text, bert_predict=weak)["status"], UNCERTAIN_CLASSIFICATION)
        self.assertEqual(validate_input(text)["status"], UNCERTAIN_CLASSIFICATION)

    def test_statuses_are_not_bert_labels(self):
        for s in iv.ALL_STATUSES:
            self.assertNotIn(s, ("DV", "SH", "ST", "CA", "WH", "OV"))


class TestBertModelPreserved(unittest.TestCase):
    MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bert_multilabel_experiment_v2"

    def test_six_labels_and_thresholds_unchanged(self):
        thr = json.loads((self.MODEL_DIR / "optimal_thresholds.json").read_text(encoding="utf-8"))
        expected = {"DV": 0.73, "SH": 0.50, "ST": 0.79, "CA": 0.45, "WH": 0.29, "OV": 0.63}
        for k, v in expected.items():
            self.assertAlmostEqual(float(thr[k]), v, places=2)
        mapping = json.loads((self.MODEL_DIR / "label_mapping.json").read_text(encoding="utf-8"))
        labels = mapping.get("labels") or list(mapping.get("label2id", {}).keys())
        self.assertEqual(sorted(labels), sorted(expected.keys()))


class TestInputValidationAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.database import SessionLocal
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.session_ids = []

    @classmethod
    def tearDownClass(cls):
        from app.models.session import Session as UserSession
        from app.models.incident_submission import IncidentSubmission
        from app.models.report import Report
        from app.models.question import SubmissionQuestionAnswer
        try:
            for sid in cls.session_ids:
                sub = cls.db.query(IncidentSubmission).filter(IncidentSubmission.session_id == sid).first()
                if sub:
                    cls.db.query(Report).filter(Report.submission_id == sub.submission_id).delete()
                    cls.db.query(SubmissionQuestionAnswer).filter(
                        SubmissionQuestionAnswer.submission_id == sub.submission_id).delete()
                    cls.db.query(IncidentSubmission).filter(
                        IncidentSubmission.submission_id == sub.submission_id).delete()
                cls.db.query(UserSession).filter(UserSession.session_id == sid).delete()
            cls.db.commit()
        except Exception:
            cls.db.rollback()
        finally:
            cls.db.close()

    def _session(self):
        from app.services.session_service import SessionService
        s = SessionService.create_session(self.db)
        self.session_ids.append(s.session_id)
        return s.session_id

    def _post(self, sid, msg):
        return self.client.post("/api/chat/message", json={"session_id": sid, "message": msg})

    def _assert_not_analysed(self, sid, msg, expected_status):
        from app.models.incident_submission import IncidentSubmission
        with mock.patch("app.services.chat_service.process_submission") as qe, \
             mock.patch("app.services.chat_service.generate_incident_report") as rep:
            res = self._post(sid, msg)
            qe.assert_not_called()
            rep.assert_not_called()
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d["input_status"], expected_status)
        self.assertEqual(d["predicted_labels"], [])
        self.assertEqual(d["entities"], {})
        self.assertIsNone(d["current_question"])
        self.assertIsNone(d["report"])
        self.assertNotEqual(d["conversation_status"], "COMPLETED")
        self.db.expire_all()
        sub = self.db.query(IncidentSubmission).filter(IncidentSubmission.session_id == sid).first()
        # Sessions pre-create an empty submission; it must remain empty (no narrative / labels).
        if sub is not None:
            self.assertFalse((sub.narrative_text or "").strip())
            labels = sub.predicted_labels
            self.assertFalse(labels.get("labels") if isinstance(labels, dict) else labels)
        # Report endpoint must not fabricate a report for a non-incident session
        rr = self.client.get(f"/api/report/{d['submission_id']}")
        self.assertEqual(rr.status_code, 404)
        return d

    def test_A_api_out_of_scope(self):
        d = self._assert_not_analysed(self._session(), CASE_A, OUT_OF_SCOPE)
        self.assertEqual(d["assistant_message"], OOS_MSG)

    def test_B_api_greeting(self):
        self._assert_not_analysed(self._session(), CASE_B, OUT_OF_SCOPE)

    def test_C_api_empty(self):
        res = self._post(self._session(), "   ")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["detail"]["input_status"], INVALID_INPUT)
        d = self._assert_not_analysed(self._session(), "?!", INVALID_INPUT)
        self.assertTrue(d["assistant_message"])

    def test_D_api_insufficient(self):
        d = self._assert_not_analysed(self._session(), CASE_D, INSUFFICIENT_INFORMATION)
        self.assertEqual(d["assistant_message"], INSUF_MSG)

    def test_H_api_uncertain(self):
        d = self._assert_not_analysed(self._session(), CASE_H, UNCERTAIN_CLASSIFICATION)
        self.assertEqual(d["assistant_message"], UNCERT_MSG)

    def test_E_F_G_api_valid_runs_original_pipeline(self):
        from app.services import chat_service
        for msg in (CASE_E, CASE_F, CASE_G):
            sid = self._session()
            with mock.patch.object(chat_service, "process_submission",
                                   wraps=chat_service.process_submission) as qe:
                res = self._post(sid, msg)
                qe.assert_called_once()
            self.assertEqual(res.status_code, 200)
            d = res.json()
            self.assertEqual(d["input_status"], VALID_INCIDENT, msg)
            self.assertIn(d["conversation_status"], ("QUESTIONING", "COMPLETED"))
            if chat_service.ChatService.get_bert_predictor() is not None:
                self.assertTrue(len(d["predicted_labels"]) > 0, msg)

    def test_expected_bert_labels_for_valid_cases(self):
        from app.services import chat_service
        if chat_service.ChatService.get_bert_predictor() is None:
            self.skipTest("BERT model not available")
        expected = {CASE_E: "Stalking", CASE_G: "Domestic Violence"}
        for msg, label in expected.items():
            d = self._post(self._session(), msg).json()
            self.assertIn(label, d["predicted_labels"], msg)

    def test_insufficient_then_details_becomes_valid(self):
        sid = self._session()
        d1 = self._post(sid, CASE_D).json()
        self.assertEqual(d1["input_status"], INSUFFICIENT_INFORMATION)
        d2 = self._post(sid, "My neighbour keeps following me and threatened me last night.").json()
        self.assertEqual(d2["input_status"], VALID_INCIDENT)
        self.assertIn(d2["conversation_status"], ("QUESTIONING", "COMPLETED"))

    def test_greeting_then_incident_excludes_greeting_from_narrative(self):
        from app.models.incident_submission import IncidentSubmission
        sid = self._session()
        self._post(sid, CASE_B)
        d = self._post(sid, CASE_G).json()
        self.assertEqual(d["input_status"], VALID_INCIDENT)
        self.db.expire_all()
        sub = self.db.query(IncidentSubmission).filter(IncidentSubmission.session_id == sid).first()
        self.assertIsNotNone(sub)
        self.assertEqual(sub.narrative_text, CASE_G)

    def test_followup_answers_not_revalidated(self):
        sid = self._session()
        d1 = self._post(sid, CASE_G).json()
        if d1["conversation_status"] != "QUESTIONING":
            self.skipTest("No follow-up question generated")
        d2 = self._post(sid, "Delhi").json()
        self.assertEqual(d2["input_status"], VALID_INCIDENT)


if __name__ == "__main__":
    unittest.main()
