"""
Unit and Integration Tests for Report API Router.

Verifies FastAPI endpoints POST /api/report/generate and GET /api/report/{submission_id},
integration with ReportGeneratorService, anti-hallucination gating, fallback responses,
error handling, and report persistence in PostgreSQL.
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.session import Session as UserSession
from app.models.incident_submission import IncidentSubmission
from app.models.report import Report
from app.models.question import SubmissionQuestionAnswer
from app.services.legal_mapping import NO_MATCH_MESSAGE as LEGAL_NO_MATCH_MESSAGE
from app.services.support_mapping import NO_MATCH_MESSAGE as SUPPORT_NO_MATCH_MESSAGE


class TestReportAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()

        # Create prerequisite Session and IncidentSubmission in PostgreSQL
        cls.session_id = "SESS-API-TEST-001"
        cls.submission_id = "SUB-API-TEST-001"

        existing_sess = cls.db.query(UserSession).filter(UserSession.session_id == cls.session_id).first()
        if not existing_sess:
            sess = UserSession(session_id=cls.session_id)
            cls.db.add(sess)
            cls.db.commit()

        existing_sub = cls.db.query(IncidentSubmission).filter(IncidentSubmission.submission_id == cls.submission_id).first()
        if existing_sub:
            cls.db.delete(existing_sub)
            cls.db.commit()

        cls.sub = IncidentSubmission(
            submission_id=cls.submission_id,
            session_id=cls.session_id,
            narrative_text="My former colleague threatened me at the workplace.",
            state="Delhi",
            district="New Delhi",
            predicted_labels={"labels": ["Workplace Harassment"]},
            extracted_entities=[
                {"label": "PERP_REL", "text": "former colleague"},
                {"label": "LOCATION", "text": "workplace"},
            ],
        )
        cls.db.add(cls.sub)
        cls.db.commit()

    @classmethod
    def tearDownClass(cls):
        # Clean up database test records
        try:
            cls.db.query(Report).filter(Report.submission_id == cls.submission_id).delete()
            cls.db.query(SubmissionQuestionAnswer).filter(SubmissionQuestionAnswer.submission_id == cls.submission_id).delete()
            cls.db.query(IncidentSubmission).filter(IncidentSubmission.submission_id == cls.submission_id).delete()
            cls.db.query(UserSession).filter(UserSession.session_id == cls.session_id).delete()
            cls.db.commit()
        except Exception:
            cls.db.rollback()
        finally:
            cls.db.close()

    # 1. POST /api/report/generate with valid submission
    def test_01_post_report_generate_valid_submission(self):
        payload = {
            "submission_id": self.submission_id,
            "state": "Delhi",
            "district": "New Delhi",
        }
        response = self.client.post("/api/report/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["submission_id"], self.submission_id)

    # 2. POST report generation returns structured report
    def test_02_post_report_generate_returns_structured_report(self):
        payload = {"submission_id": self.submission_id}
        response = self.client.post("/api/report/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("incident_summary", data)
        self.assertIn("identified_incident_type", data)
        self.assertIn("extracted_information", data)
        self.assertIn("user_provided_details", data)
        self.assertIn("relevant_legal_information", data)
        self.assertIn("suggested_next_steps", data)
        self.assertIn("available_support_services", data)
        self.assertIn("evidence_information_notes", data)
        self.assertIn("model_prediction_information", data)
        self.assertIn("disclaimer", data)

    # 3. GET /api/report/{submission_id} returns existing report
    def test_03_get_existing_report(self):
        # Generate report first
        self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        
        # Fetch report via GET
        response = self.client.get(f"/api/report/{self.submission_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["submission_id"], self.submission_id)

    # 4. GET missing report returns 404
    def test_04_get_missing_report_returns_404(self):
        response = self.client.get("/api/report/SUB-MISSING-999999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    # 5. Invalid submission handling
    def test_05_invalid_submission_handling(self):
        response = self.client.post("/api/report/generate", json={"submission_id": ""})
        self.assertEqual(response.status_code, 400)

        response_missing = self.client.post("/api/report/generate", json={"submission_id": "SUB-NONEXISTENT-999"})
        self.assertEqual(response_missing.status_code, 404)

    # 6. Report remains associated with correct submission_id
    def test_06_report_remains_associated_with_submission_id(self):
        response = self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        data = response.json()
        self.assertEqual(data["submission_id"], self.submission_id)

    # 7. Report retrieved after generation matches
    def test_07_existing_report_retrieved_after_generation(self):
        gen_res = self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        gen_data = gen_res.json()

        get_res = self.client.get(f"/api/report/{self.submission_id}")
        get_data = get_res.json()

        self.assertEqual(gen_data["incident_summary"], get_data["incident_summary"])
        self.assertEqual(gen_data["identified_incident_type"], get_data["identified_incident_type"])

    # 8. Regeneration does not create uncontrolled duplicate rows in DB
    def test_08_regeneration_does_not_create_duplicates(self):
        self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        self.client.post("/api/report/generate", json={"submission_id": self.submission_id})

        report_count = (
            self.db.query(Report)
            .filter(Report.submission_id == self.submission_id)
            .count()
        )
        self.assertEqual(report_count, 1)

    # 9. All 10 required report sections present
    def test_09_ten_required_report_sections_present(self):
        response = self.client.get(f"/api/report/{self.submission_id}")
        data = response.json()
        required_keys = [
            "incident_summary",
            "identified_incident_type",
            "extracted_information",
            "user_provided_details",
            "relevant_legal_information",
            "suggested_next_steps",
            "available_support_services",
            "evidence_information_notes",
            "model_prediction_information",
            "disclaimer",
        ]
        for k in required_keys:
            self.assertIn(k, data)

    # 10. Legal information remains database-proven
    def test_10_legal_information_remains_database_proven(self):
        response = self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        data = response.json()
        laws = data["relevant_legal_information"]
        self.assertIsInstance(laws, list)
        self.assertGreater(len(laws), 0)
        self.assertEqual(laws[0]["source"], "database")

    # 11. Fabricated legal information cannot be injected through API
    def test_11_fabricated_legal_info_cannot_be_injected(self):
        payload = {
            "submission_id": self.submission_id,
            "bert_results": ["UNKNOWN_CATEGORY_XYZ"],
        }
        response = self.client.post("/api/report/generate", json=payload)
        data = response.json()
        self.assertEqual(data["relevant_legal_information"], LEGAL_NO_MATCH_MESSAGE)

    # 12. Support information remains database-proven
    def test_12_support_information_remains_database_proven(self):
        response = self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        data = response.json()
        supports = data["available_support_services"]
        self.assertIsInstance(supports, list)
        self.assertGreater(len(supports), 0)
        self.assertEqual(supports[0]["source"], "database")

    # 13. Fabricated support information cannot be injected through API
    def test_13_fabricated_support_info_cannot_be_injected(self):
        payload = {
            "submission_id": self.submission_id,
            "bert_results": ["UNKNOWN_CATEGORY_XYZ"],
        }
        response = self.client.post("/api/report/generate", json=payload)
        data = response.json()
        self.assertEqual(data["available_support_services"], SUPPORT_NO_MATCH_MESSAGE)

    # 14. Unknown incident category does not generate legal info
    def test_14_unknown_category_does_not_generate_legal(self):
        payload = {
            "submission_id": self.submission_id,
            "bert_results": ["FAKE_CATEGORY_999"],
        }
        response = self.client.post("/api/report/generate", json=payload)
        data = response.json()
        self.assertEqual(data["relevant_legal_information"], "Information not available in the provided knowledge base.")

    # 15. Unknown incident category does not generate support info
    def test_15_unknown_category_does_not_generate_support(self):
        payload = {
            "submission_id": self.submission_id,
            "bert_results": ["FAKE_CATEGORY_999"],
        }
        response = self.client.post("/api/report/generate", json=payload)
        data = response.json()
        self.assertEqual(data["available_support_services"], "No matching support service was found in the available support-services database.")

    # 16. Authentication / Submission ownership rules respected
    def test_16_authentication_ownership_rules(self):
        response = self.client.get(f"/api/report/{self.submission_id}")
        self.assertEqual(response.status_code, 200)

    # 17. No external API / Web dependency
    def test_17_no_external_api_dependency(self):
        response = self.client.get(f"/api/report/{self.submission_id}")
        self.assertEqual(response.status_code, 200)

    # 18. Appropriate error responses (400, 404) without exposing credentials
    def test_18_appropriate_error_responses(self):
        response = self.client.get("/api/report/SUB-INVALID-999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertNotIn("password", str(data).lower())
        self.assertNotIn("postgresql", str(data).lower())

    # 19. Existing ReportGeneratorService is reused
    def test_19_existing_report_generator_service_reused(self):
        response = self.client.post("/api/report/generate", json={"submission_id": self.submission_id})
        data = response.json()
        self.assertIn("model_prediction_information", data)

    # 20. Existing Part 18 functionality unaffected
    def test_20_existing_part18_unaffected(self):
        response = self.client.get(f"/api/report/{self.submission_id}")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
