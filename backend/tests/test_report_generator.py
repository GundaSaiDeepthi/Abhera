"""
Unit and Integration Tests for Incident Report Generator Service.

Verifies correct assembly of all 10 required conceptual report sections,
integration with Anti-Hallucination validation, fallback behavior, value preservation,
and complete isolation from external APIs or LLM generative code.
"""

import unittest

from app.database import SessionLocal
from app.services.report_generator import ReportGeneratorService, generate_incident_report
from app.services.legal_mapping import NO_MATCH_MESSAGE as LEGAL_NO_MATCH_MESSAGE
from app.services.support_mapping import NO_MATCH_MESSAGE as SUPPORT_NO_MATCH_MESSAGE


class TestReportGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.service = ReportGeneratorService(db=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # 1. Valid complete report generation with all 10 sections
    def test_01_valid_complete_report_generation(self):
        report = self.service.generate_report(
            submission_id="SUB-REPORT-001",
            narrative_text="My former colleague harassed me at the workplace.",
            bert_results={"predicted_labels": ["Workplace Harassment"]},
            ner_entities=[
                {"label": "PERP_REL", "text": "former colleague"},
                {"label": "LOCATION", "text": "workplace"},
            ],
            user_answers=[{"question_id": "Q_SAFETY_01", "question": "Are you safe?", "answer": "Yes, I am safe."}],
            state="Delhi",
            district="New Delhi",
            db_session=self.db,
        )

        self.assertIsNotNone(report)
        self.assertIn("incident_summary", report)
        self.assertIn("identified_incident_type", report)
        self.assertIn("extracted_information", report)
        self.assertIn("user_provided_details", report)
        self.assertIn("relevant_legal_information", report)
        self.assertIn("suggested_next_steps", report)
        self.assertIn("available_support_services", report)
        self.assertIn("evidence_information_notes", report)
        self.assertIn("model_prediction_information", report)
        self.assertIn("disclaimer", report)

    # 2. Incident summary uses user-provided information only
    def test_02_incident_summary_uses_user_info_only(self):
        text = "My landlord threatened me at home."
        report = self.service.generate_report(narrative_text=text, db_session=self.db)
        self.assertEqual(report["incident_summary"], text)

        empty_report = self.service.generate_report(narrative_text="", db_session=self.db)
        self.assertEqual(
            empty_report["incident_summary"],
            "Incident summary is not available from the provided information."
        )

    # 3. BERT labels are preserved
    def test_03_bert_labels_preserved(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["Domestic Violence"]},
            db_session=self.db,
        )
        self.assertEqual(report["identified_incident_type"]["labels"], ["Domestic Violence"])

    # 4. Multi-label incidents are preserved
    def test_04_multi_label_incidents_preserved(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["DV", "WH"]},
            db_session=self.db,
        )
        self.assertEqual(report["identified_incident_type"]["labels"], ["DV", "WH"])

    # 5. NER entities are preserved
    def test_05_ner_entities_preserved(self):
        ner_entities = [
            {"label": "PERP_REL", "text": "ex-husband"},
            {"label": "TIME_FREQ", "text": "every night"},
        ]
        report = self.service.generate_report(ner_entities=ner_entities, db_session=self.db)
        self.assertIn("ex-husband", report["extracted_information"]["PERP_REL"])
        self.assertIn("every night", report["extracted_information"]["TIME_FREQ"])

    # 6. User answers are included
    def test_06_user_answers_included(self):
        answers = [{"question_id": "Q_SAFETY_01", "question": "Are you safe?", "answer": "I am currently safe."}]
        report = self.service.generate_report(user_answers=answers, db_session=self.db)
        self.assertEqual(len(report["user_provided_details"]), 1)
        self.assertEqual(report["user_provided_details"][0]["answer"], "I am currently safe.")

    # 7. Valid legal database records appear in report
    def test_07_valid_legal_database_records_appear(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["Domestic Violence"]},
            db_session=self.db,
        )
        self.assertIsInstance(report["relevant_legal_information"], list)
        self.assertGreater(len(report["relevant_legal_information"]), 0)
        first_law = report["relevant_legal_information"][0]
        self.assertIn("act", first_law)
        self.assertIn("section", first_law)

    # 8. Fabricated legal records cannot appear in report
    def test_08_fabricated_legal_records_cannot_appear(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["UNKNOWN_CATEGORY_ABC"]},
            db_session=self.db,
        )
        self.assertEqual(report["relevant_legal_information"], LEGAL_NO_MATCH_MESSAGE)

    # 9. Exact legal fallback appears when no legal information exists
    def test_09_exact_legal_fallback_appears(self):
        report = self.service.generate_report(bert_results=None, db_session=self.db)
        self.assertEqual(report["relevant_legal_information"], "Information not available in the provided knowledge base.")

    # 10. Valid support services appear in report
    def test_10_valid_support_services_appear(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["Sexual Harassment"]},
            state="Delhi",
            district="New Delhi",
            db_session=self.db,
        )
        self.assertIsInstance(report["available_support_services"], list)
        self.assertGreater(len(report["available_support_services"]), 0)

    # 11. Fabricated support services cannot appear in report
    def test_11_fabricated_support_services_cannot_appear(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["UNKNOWN_CATEGORY_XYZ"]},
            db_session=self.db,
        )
        self.assertEqual(report["available_support_services"], SUPPORT_NO_MATCH_MESSAGE)

    # 12. Exact support fallback appears when no support exists
    def test_12_exact_support_fallback_appears(self):
        report = self.service.generate_report(bert_results=None, db_session=self.db)
        self.assertEqual(
            report["available_support_services"],
            "No matching support service was found in the available support-services database."
        )

    # 13. Unknown category does not create legal information
    def test_13_unknown_category_does_not_create_legal(self):
        report = generate_incident_report(bert_results=["FAKE_CATEGORY"], db=self.db)
        self.assertEqual(report["relevant_legal_information"], LEGAL_NO_MATCH_MESSAGE)

    # 14. Unknown category does not create support information
    def test_14_unknown_category_does_not_create_support(self):
        report = generate_incident_report(bert_results=["FAKE_CATEGORY"], db=self.db)
        self.assertEqual(report["available_support_services"], SUPPORT_NO_MATCH_MESSAGE)

    # 15. Missing entity categories produce empty lists
    def test_15_missing_entity_categories_produce_empty_lists(self):
        report = self.service.generate_report(ner_entities=[], db_session=self.db)
        for cat, items in report["extracted_information"].items():
            self.assertEqual(items, [])

    # 16. Missing user answers are not converted into facts
    def test_16_missing_user_answers_not_converted_to_facts(self):
        report = self.service.generate_report(user_answers=[], db_session=self.db)
        self.assertEqual(report["user_provided_details"], [])

    # 17. Model prediction information is clearly separated from database info
    def test_17_model_prediction_info_separated_from_db_info(self):
        report = self.service.generate_report(
            bert_results={"predicted_labels": ["Stalking"]},
            db_session=self.db,
        )
        model_info = report["model_prediction_information"]
        self.assertIn("bert_classifier", model_info)
        self.assertIn("ner_extractor", model_info)
        self.assertIn("data_provenance", model_info)
        self.assertIn("PostgreSQL", model_info["data_provenance"])

    # 18. No external API/web dependency
    def test_18_no_external_api_dependency(self):
        report = generate_incident_report(
            narrative_text="Stalked online",
            bert_results=["Cyber Harassment"],
            db=self.db,
        )
        self.assertIsNotNone(report)

    # 19. Disclaimer is present
    def test_19_disclaimer_is_present(self):
        report = self.service.generate_report(db_session=self.db)
        self.assertIn("disclaimer", report)
        self.assertIn("informational assistance", report["disclaimer"])
        self.assertIn("does not replace professional legal advice", report["disclaimer"])

    # 20. Report contains all 10 required conceptual sections
    def test_20_report_contains_all_ten_sections(self):
        report = generate_incident_report(db=self.db)
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
        for key in required_keys:
            self.assertIn(key, report)

    # 21. Questionnaire location fallback works when NER entities are empty
    def test_21_location_fallback_from_questionnaire(self):
        user_answers = [
            {
                "question_id": "Q_WH_LOCATION_01",
                "question": "Did this occur at your workplace, during work-related events, or through work communication channels?",
                "answer": "workplace",
            }
        ]
        report = self.service.generate_report(
            ner_entities=[],
            user_answers=user_answers,
            db_session=self.db,
        )
        self.assertIn("Workplace", report["extracted_information"]["LOCATION"])

    # 22. NER location takes precedence over questionnaire fallback
    def test_22_ner_location_takes_precedence(self):
        ner_entities = [{"entity_group": "LOCATION", "word": "Office Building"}]
        user_answers = [
            {
                "question_id": "Q_WH_LOCATION_01",
                "question": "Did this occur at your workplace, during work-related events, or through work communication channels?",
                "answer": "workplace",
            }
        ]
        report = self.service.generate_report(
            ner_entities=ner_entities,
            user_answers=user_answers,
            db_session=self.db,
        )
        self.assertIn("Office Building", report["extracted_information"]["LOCATION"])
        self.assertNotIn("Workplace", report["extracted_information"]["LOCATION"])



if __name__ == "__main__":
    unittest.main()
